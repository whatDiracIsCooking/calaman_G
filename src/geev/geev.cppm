/**
 * @file geev.cppm
 * @brief The calaman.geev module -- eigenvalues and optional left/right
 *        eigenvectors of a general real matrix, LAPACK's ?geev
 *
 * Computes the eigenvalues (@p wr + i*@p wi) of a real n-by-n column-major
 * matrix @p a and, when asked, its right (@p vr) and/or left (@p vl)
 * eigenvectors. REAL ONLY (float / double), the scope ?sgeev / ?dgeev has.
 *
 * The full ?geev DRIVER as a HOST COMPOSITION in calaman.hseqr's shape: the O(1)
 * control flow (the scaling decision, the ilo/ihi gebal reports, the per-vector
 * normalization) runs on the host and stages the scalars it inspects through
 * device memcpies, while every heavy step is a shipped kernel it dispatches to.
 * The chain is ?geev's exactly: optional calaman.lange/calaman.lascl scaling of
 * an out-of-range @p a, calaman.gebal balancing, calaman.gehrd reduction to upper
 * Hessenberg form, then -- for vectors -- calaman.orghr to form Q and
 * calaman.hseqr (Schur form, accumulating Q) else calaman.hseqr for eigenvalues
 * only, calaman.trevc3 for the eigenvectors of the Schur form, calaman.gebak to
 * undo the balancing, and a final unit-2-norm + largest-real normalization.
 *
 * ONE DELIBERATE DEVIATION vs the reference: ?dgeev calls ?trevc3 with
 * HOWMNY = 'B' so the back-substitution back-transforms through the Schur vectors
 * in place; calaman.trevc3 is HOWMNY = 'A' only (eigenvectors of the Schur form
 * T, no OVER step). So geev computes X = eigenvectors of T into scratch and forms
 * V = Q*X itself with one gemm per side -- the same arithmetic, a separate buffer.
 *
 * Allocation-free shipped surface (CLAUDE.md): @p a, @p wr, @p wi, @p vl, @p vr,
 * @p work and @p info are caller device pointers; @p work is one buffer of
 * geev_bufferSize<T>() bytes. SYNCHRONIZES: the driver branches on device data.
 * Both handles must be in the DEFAULT (host) pointer mode and set to the SAME
 * stream (geev reads it off @p blas); the solver handle backs only calaman.orghr.
 *
 * Usage:
 *   import calaman.geev;
 *   import wwr.blas; import wwr.solver;
 *   const std::size_t bytes =
 *       calaman::geev_bufferSize<double>(solver, n, GeevVectors::Vectors,
 *                                        GeevVectors::Vectors);
 *   calaman::geev<double>(blas, solver, GeevVectors::Vectors, GeevVectors::Vectors,
 *                         n, d_a, lda, d_wr, d_wi, d_vl, ldvl, d_vr, ldvr,
 *                         d_work, bytes, d_info);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the export import
// below supplies.
#include "error_handling/error_macros.h"

export module calaman.geev;

import std;
import wwr.blas;            // wwrblasHandle_t, wwrblasGetStream, WWRBLAS_OP_N, status
import wwr.solver;          // wwrsolverDnHandle_t (calaman.orghr's handle)
import wwr.runtime_api;     // wwrStream_t, wwrMemcpyAsync, wwrStreamSynchronize
import wwr.wrappers.blas;   // gemm (back-transform), nrm2 / scal / rot (normalize)
import wwr.wrappers.common; // real_fp
import calaman.common;      // Region, MatrixNorm, WorkspaceLayout
import calaman.lange;       // lange -- the max-abs norm for the scaling decision
import calaman.lascl;       // lascl -- scale a, rescale the eigenvalues
import calaman.lacpy;       // lacpy -- copy the gehrd reflectors into Q
import calaman.gebal;       // gebal + GebalJob -- balancing
import calaman.gehrd;       // gehrd -- Hessenberg reduction
import calaman.orghr;       // orghr + orghr_bufferSize -- form Q
import calaman.hseqr;       // hseqr + HseqrJob / HseqrCompz -- Schur form / eigenvalues
import calaman.trevc3;      // trevc3 -- eigenvectors of the Schur form
import calaman.gebak;       // gebak + GebakJob / GebakSide -- undo balancing

// export import, not a plain import: geev RETURNS calaman::Status, so a consumer
// of `import calaman.geev;` must see Status's member functions, not just its name
// -- the same re-export the chained modules do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

/// @brief Whether a side's eigenvectors are computed -- LAPACK's JOBVL / JOBVR
///
/// None ('N'): that side is not referenced. Vectors ('V'): compute and normalize
/// that side's eigenvectors.
export enum class GeevVectors { None, Vectors };

namespace geev_detail {

// Internal gehrd block size: calaman.gehrd's kGehrdBlockSize is not exported, so
// its work length (n*nb + nb*nb) is reproduced here, as the gehrd oracle does.
inline constexpr int kGeevGehrdNb = 32;

// The hseqr lwork governing laqr0's NWMAX / NSMAX; the value calaman.hseqr's
// oracle passes (6n + 16 >= 2(n-1)/3, so the window/shift schedules match).
inline int hseqr_lwork(const int n) { return 6 * n + 16; }

// laqr0's recursive-laqr4 scratch length for a subproblem of order @p sub at
// @p lwork -- the tail of the carve calaman.hseqr forwards to calaman.laqr0.
inline std::size_t laqr4_workbuf_len(const int sub, const int lwork) {
  const int nwmax = std::min((sub - 1) / 3, lwork / 2);
  int nsmax = std::min((sub - 3) / 6, 2 * lwork / 3);
  nsmax -= nsmax % 2;
  const int ldv = std::max(1, nwmax);
  const int ldsc = std::max(1, nsmax);
  return 2 * static_cast<std::size_t>(ldv) * std::max(1, nwmax) +
         static_cast<std::size_t>(ldsc) * std::max(1, nsmax) + 4;
}

// The whole work length calaman.hseqr requires for order @p n at @p lwork -- the
// laqr0 carve (2 V/T windows + NS-by-NS shift copy + Z dummy + recursive laqr4
// scratch + device counters). Reproduced from laqr0's header; hseqr forwards it
// unchanged and never reads it on its small-n lahqr path.
inline std::size_t hseqr_workbuf_len(const int n, const int lwork) {
  const int nwmax = std::min((n - 1) / 3, lwork / 2);
  int nsmax = std::min((n - 3) / 6, 2 * lwork / 3);
  nsmax -= nsmax % 2;
  const int ldv = std::max(1, nwmax);
  const int ldsc = std::max(1, nsmax);
  return 2 * static_cast<std::size_t>(ldv) * std::max(1, nwmax) +
         static_cast<std::size_t>(ldsc) * std::max(1, nsmax) + 1 +
         laqr4_workbuf_len(nwmax, lwork) + 4;
}

/// @brief geev's device workspace: the regions that must stay live, plus the
///        shared scratch zone every transient step aliases
template<typename T>
struct GeevWorkspace {
  T *anrm;             ///< lange's max-abs result scalar
  T *scale;            ///< gebal's scale (length n), live through gebak
  T *tau;              ///< gehrd/orghr reflector scalars (length n)
  T *q;                ///< Schur vectors (n x n), null when no side is wanted
  T *xl;               ///< left eigenvectors of T (n x n), null unless wantvl
  T *xr;               ///< right eigenvectors of T (n x n), null unless wantvr
  void *scratch;       ///< base of the shared scratch zone (gebal/gehrd/orghr/hseqr/trevc3)
  std::size_t orghr_bytes; ///< orghr's own byte budget inside the scratch zone
  std::size_t total;   ///< bytes the layout spans
};

/// @brief Carve geev's workspace, SIZING over @p base == nullptr or carving it
///
/// The fixed regions coexist (scale lives from gebal to gebak, Q from orghr to
/// the back-transform); the five scratch users are never live at once, so they
/// share one zone sized to the largest. orghr's window budget is taken at the
/// widest window gebal can leave (ilo = 1, ihi = n), a safe upper bound.
template<typename T>
GeevWorkspace<T> map_workspace(wwr::wwrsolverDnHandle_t solver, void *base, const int n,
                               const bool wantvl, const bool wantvr) {
  const std::size_t nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  const std::size_t nsz = static_cast<std::size_t>(n < 1 ? 1 : n);
  const bool wantv = wantvl || wantvr;
  const std::size_t orghr_bytes =
      wantv ? orghr_bufferSize<T>(solver, n, n, 1, n) : std::size_t{0};

  WorkspaceLayout layout(base);
  GeevWorkspace<T> ws{};
  ws.anrm = layout.fixed<T>(1);
  ws.scale = layout.fixed<T>(nsz);
  ws.tau = layout.fixed<T>(nsz);
  ws.q = wantv ? layout.fixed<T>(nn) : nullptr;
  ws.xl = wantvl ? layout.fixed<T>(nn) : nullptr;
  ws.xr = wantvr ? layout.fixed<T>(nn) : nullptr;

  // Scratch (declared after every fixed region, as WorkspaceLayout requires).
  const std::size_t gebal_ints = static_cast<std::size_t>(2 * n + 1);
  const std::size_t gehrd_len = static_cast<std::size_t>(n) * kGeevGehrdNb +
                                static_cast<std::size_t>(kGeevGehrdNb) * kGeevGehrdNb;
  const std::size_t hseqr_len = hseqr_workbuf_len(n, hseqr_lwork(n));
  (void)layout.scratch<int>(gebal_ints);
  (void)layout.scratch<T>(gehrd_len);
  (void)layout.scratch<T>(hseqr_len);
  if (wantv) {
    (void)layout.scratch<std::byte>(orghr_bytes);
    (void)layout.scratch<T>(static_cast<std::size_t>(3) * nsz); // trevc3
  }
  ws.scratch = layout.scratch<std::byte>(0); // the shared zone's base
  ws.orghr_bytes = orghr_bytes;
  ws.total = layout.total();
  return ws;
}

/// @brief Normalize each eigenvector of one side in place (?geev's final pass)
///
/// Real columns (@p wi == 0) scale to unit 2-norm; a complex pair scales together
/// to unit 2-norm, then a Givens rotation (?lartg + ?rot) makes its
/// largest-modulus component real. @p wi is read to the host for the block
/// structure (its zero/sign is scaling-invariant). Host-driven, host pointer mode.
template<wwr::real_fp T>
Status normalize_side(wwr::wwrblasHandle_t blas, wwr::wwrStream_t stream, const int n,
                      const T *wi_dev, T *v, const int ldv) {
  using std::size_t;
  std::vector<T> wi(static_cast<size_t>(n));
  CLM_TRY(wwr::wwrMemcpyAsync(wi.data(), wi_dev, static_cast<size_t>(n) * sizeof(T),
                              wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));

  const T one{1};
  for (int i = 0; i < n;) {
    T *const col_i = v + static_cast<size_t>(i) * static_cast<size_t>(ldv);
    if (wi[static_cast<size_t>(i)] == T{0}) {
      T nrm{};
      CLM_TRY(wwr::nrm2<T>(blas, n, col_i, 1, &nrm));
      if (nrm > T{0}) {
        const T scl = one / nrm;
        CLM_TRY(wwr::scal<T>(blas, n, &scl, col_i, 1));
      }
      i += 1;
      continue;
    }
    // Complex-conjugate pair: column i the real part, i+1 the imaginary part.
    T *const col_j = v + static_cast<size_t>(i + 1) * static_cast<size_t>(ldv);
    T n1{}, n2{};
    CLM_TRY(wwr::nrm2<T>(blas, n, col_i, 1, &n1));
    CLM_TRY(wwr::nrm2<T>(blas, n, col_j, 1, &n2));
    const T denom = std::hypot(n1, n2);
    if (denom > T{0}) {
      const T scl = one / denom;
      CLM_TRY(wwr::scal<T>(blas, n, &scl, col_i, 1));
      CLM_TRY(wwr::scal<T>(blas, n, &scl, col_j, 1));
    }
    // Largest-modulus row of the pair, then the rotation zeroing its imag part.
    std::vector<T> ci(static_cast<size_t>(n)), cj(static_cast<size_t>(n));
    CLM_TRY(wwr::wwrMemcpyAsync(ci.data(), col_i, static_cast<size_t>(n) * sizeof(T),
                                wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(wwr::wwrMemcpyAsync(cj.data(), col_j, static_cast<size_t>(n) * sizeof(T),
                                wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    int k = 0;
    T best = T{-1};
    for (int r = 0; r < n; ++r) {
      const T mag = ci[static_cast<size_t>(r)] * ci[static_cast<size_t>(r)] +
                    cj[static_cast<size_t>(r)] * cj[static_cast<size_t>(r)];
      if (mag > best) {
        best = mag;
        k = r;
      }
    }
    const T rr = std::hypot(ci[static_cast<size_t>(k)], cj[static_cast<size_t>(k)]);
    if (rr > T{0}) {
      const T cs = ci[static_cast<size_t>(k)] / rr;
      const T sn = cj[static_cast<size_t>(k)] / rr;
      CLM_TRY(wwr::rot<T>(blas, n, col_i, 1, col_j, 1, &cs, &sn));
      const T z{0}; // pin the now-real component: clear the residual imag entry
      CLM_TRY(wwr::wwrMemcpyAsync(col_j + static_cast<size_t>(k), &z, sizeof(T),
                                  wwr::wwrMemcpyHostToDevice, stream));
      CLM_TRY(wwr::wwrStreamSynchronize(stream));
    }
    i += 2;
  }
  return wwr::wwrSuccess;
}

} // namespace geev_detail

// ========================================================================
// Workspace size (exported)
// ========================================================================

/// @brief Device workspace geev() needs, in bytes
///
/// Covers gebal's scale, the gehrd/orghr reflector scalars, the Schur-vector and
/// eigenvector-of-T matrices (only the sides @p jobvl / @p jobvr request), and
/// the shared scratch zone the staged kernels reuse. @p solver sizes orghr's
/// slice; pass the same handle to geev().
///
/// @tparam T     Element type (float or double)
/// @param solver Solver handle, used only to size orghr's workspace
/// @param n      Order of the matrix
/// @param jobvl  Whether left eigenvectors are wanted
/// @param jobvr  Whether right eigenvectors are wanted
export template<wwr::real_fp T>
std::size_t geev_bufferSize(wwr::wwrsolverDnHandle_t solver, const int n, const GeevVectors jobvl,
                            const GeevVectors jobvr) {
  return geev_detail::map_workspace<T>(solver, nullptr, n, jobvl == GeevVectors::Vectors,
                                       jobvr == GeevVectors::Vectors)
      .total;
}

// ========================================================================
// Driver (exported)
// ========================================================================

/// @brief Eigenvalues and optional left/right eigenvectors of a real matrix (?geev)
///
/// Runs the ?geev chain and SYNCHRONIZES: on return @p wr / @p wi hold the
/// eigenvalues, and when a side is requested its eigenvectors fill @p vl / @p vr
/// (one column per real eigenvalue, two -- real then imaginary part -- per
/// complex-conjugate pair, each normalized to unit 2-norm with its largest
/// component real), @p a is overwritten (its Schur form on the vector path). @p
/// info is 0 on success, -k for an illegal k-th argument, or the ?hseqr
/// convergence index (eigenvalues @p info+1.. are valid, eigenvectors are not).
///
/// @tparam T Element type (float, double)
/// @param blas BLAS handle, host pointer mode; its stream carries the whole run
/// @param solver Solver handle on the SAME stream, host pointer mode (orghr only)
/// @param jobvl Whether to compute left eigenvectors into @p vl
/// @param jobvr Whether to compute right eigenvectors into @p vr
/// @param n Order of @p a
/// @param a Device n-by-n matrix, column-major, leading dim @p lda; overwritten
/// @param lda Leading dimension of @p a (>= max(1, n))
/// @param wr Device length->= @p n real parts of the eigenvalues
/// @param wi Device length->= @p n imaginary parts of the eigenvalues
/// @param vl Device n-by-n left eigenvectors, leading dim @p ldvl; written if @p jobvl
/// @param ldvl Leading dimension of @p vl (>= n if @p jobvl, else >= 1)
/// @param vr Device n-by-n right eigenvectors, leading dim @p ldvr; written if @p jobvr
/// @param ldvr Leading dimension of @p vr (>= n if @p jobvr, else >= 1)
/// @param work Device buffer of >= geev_bufferSize<T>(@p solver, n, jobvl, jobvr) bytes
/// @param work_bytes Size of @p work in bytes
/// @param info Device int receiving the ?geev INFO
/// @return Success, or the runtime error a staged step reported
export template<wwr::real_fp T>
Status geev(wwr::wwrblasHandle_t blas, wwr::wwrsolverDnHandle_t solver, const GeevVectors jobvl,
            const GeevVectors jobvr, const int n, T *const a, const int lda, T *const wr,
            T *const wi, T *const vl, const int ldvl, T *const vr, const int ldvr,
            void *const work, const std::size_t work_bytes, int *const info) {
  using std::size_t;
  const bool wantvl = (jobvl == GeevVectors::Vectors);
  const bool wantvr = (jobvr == GeevVectors::Vectors);
  const bool wantv = wantvl || wantvr;

  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(blas, &stream));

  auto write_info = [&](int v) -> Status {
    CLM_TRY(wwr::wwrMemcpyAsync(info, &v, sizeof(int), wwr::wwrMemcpyHostToDevice, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    return wwr::wwrSuccess;
  };

  // ==== Argument validation, numbered by geev's own parameter positions. ====
  int ierr = 0;
  if (n < 0) {
    ierr = -5;
  } else if (lda < std::max(1, n)) {
    ierr = -7;
  } else if (ldvl < 1 || (wantvl && ldvl < n)) {
    ierr = -11;
  } else if (ldvr < 1 || (wantvr && ldvr < n)) {
    ierr = -13;
  }
  if (ierr != 0) {
    return write_info(ierr);
  }
  if (n == 0) {
    return write_info(0);
  }

  const auto ws = geev_detail::map_workspace<T>(solver, work, n, wantvl, wantvr);
  if (work == nullptr || work_bytes < ws.total) {
    return write_info(-15);
  }

  // ==== Optional scaling of a that is far out of range (?geev's SCALEA). ====
  const T eps = std::numeric_limits<T>::epsilon();
  const T smlnum = std::sqrt(std::numeric_limits<T>::min()) / eps;
  const T bignum = T{1} / smlnum;
  CLM_TRY(lange<T>(stream, MatrixNorm::max_abs, static_cast<size_t>(n), static_cast<size_t>(n), a,
                   static_cast<size_t>(lda), ws.anrm));
  T anrm{};
  CLM_TRY(wwr::wwrMemcpyAsync(&anrm, ws.anrm, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  bool scalea = false;
  T cscale = T{1};
  if (anrm > T{0} && anrm < smlnum) {
    scalea = true;
    cscale = smlnum;
  } else if (anrm > bignum) {
    scalea = true;
    cscale = bignum;
  }
  if (scalea) {
    CLM_TRY(lascl<T>(stream, anrm, cscale, static_cast<size_t>(n), static_cast<size_t>(n), a,
                     static_cast<size_t>(lda)));
  }

  // ==== Balance, then reduce to upper Hessenberg form. ====
  int ilo = 1, ihi = n;
  CLM_TRY(gebal<T>(stream, GebalJob::Both, n, a, lda, &ilo, &ihi, ws.scale,
                   reinterpret_cast<int *>(ws.scratch)));
  CLM_TRY(gehrd<T>(blas, n, ilo, ihi, a, lda, ws.tau, reinterpret_cast<T *>(ws.scratch)));

  // ==== Schur form (DHSEQR). With vectors: copy the reflectors into Q, form the
  //      explicit Q (DORGHR), then accumulate the QR rotations onto it. Else
  //      eigenvalues only. ====
  T *const hswork = reinterpret_cast<T *>(ws.scratch);
  const int hs_lwork = geev_detail::hseqr_lwork(n);
  if (wantv) {
    CLM_TRY(lacpy<T>(stream, Region::L, static_cast<size_t>(n), static_cast<size_t>(n), a,
                     static_cast<size_t>(lda), ws.q, static_cast<size_t>(n)));
    CLM_TRY(orghr<T>(solver, n, ilo, ihi, ws.q, n, ws.tau, ws.scratch, ws.orghr_bytes, info));
    CLM_TRY(hseqr<T>(stream, HseqrJob::Schur, HseqrCompz::Vectors, n, ilo, ihi, a, lda, wr, wi,
                     ws.q, n, hswork, hs_lwork, info));
  } else {
    CLM_TRY(hseqr<T>(stream, HseqrJob::Eigenvalues, HseqrCompz::None, n, ilo, ihi, a, lda, wr, wi,
                     nullptr, 1, hswork, hs_lwork, info));
  }
  int hinfo = 0;
  CLM_TRY(wwr::wwrMemcpyAsync(&hinfo, info, sizeof(int), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));

  // ==== Eigenvectors: of T (DTREVC3, HOWMNY='A'), then back-transform V = Q*X,
  //      undo the balancing, normalize. Skipped on a ?hseqr convergence failure. ====
  if (wantv && hinfo == 0) {
    T *const xl = wantvl ? ws.xl : nullptr;
    T *const xr = wantvr ? ws.xr : nullptr;
    CLM_TRY(trevc3<T>(stream, wantvl, wantvr, n, a, lda, xl, wantvl ? n : 1, xr, wantvr ? n : 1,
                      reinterpret_cast<T *>(ws.scratch)));

    const T one{1}, zero{0};
    if (wantvr) {
      CLM_TRY(wwr::gemm<T>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &one, ws.q, n, xr, n,
                           &zero, vr, ldvr));
      CLM_TRY(gebak<T>(stream, GebakJob::Both, GebakSide::Right, n, ilo, ihi, ws.scale, n, vr,
                       ldvr));
    }
    if (wantvl) {
      CLM_TRY(wwr::gemm<T>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &one, ws.q, n, xl, n,
                           &zero, vl, ldvl));
      CLM_TRY(gebak<T>(stream, GebakJob::Both, GebakSide::Left, n, ilo, ihi, ws.scale, n, vl, ldvl));
    }
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    if (wantvr) {
      CLM_TRY(geev_detail::normalize_side<T>(blas, stream, n, wi, vr, ldvr));
    }
    if (wantvl) {
      CLM_TRY(geev_detail::normalize_side<T>(blas, stream, n, wi, vl, ldvl));
    }
  }

  // ==== Undo the a-scaling on the eigenvalues (?geev's label-50 rescale). ====
  if (scalea) {
    const int m = n - hinfo;
    if (m > 0) {
      const size_t ld = static_cast<size_t>(std::max(m, 1));
      CLM_TRY(lascl<T>(stream, cscale, anrm, static_cast<size_t>(m), 1, wr + hinfo, ld));
      CLM_TRY(lascl<T>(stream, cscale, anrm, static_cast<size_t>(m), 1, wi + hinfo, ld));
    }
    if (hinfo > 0 && ilo > 1) {
      CLM_TRY(lascl<T>(stream, cscale, anrm, static_cast<size_t>(ilo - 1), 1, wr,
                       static_cast<size_t>(n)));
      CLM_TRY(lascl<T>(stream, cscale, anrm, static_cast<size_t>(ilo - 1), 1, wi,
                       static_cast<size_t>(n)));
    }
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
  }

  return write_info(hinfo);
}

} // namespace calaman
