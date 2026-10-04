/**
 * @file hseqr.cppm
 * @brief The calaman.hseqr module -- Schur factorization / eigenvalues of an
 *        upper Hessenberg matrix, dispatching calaman.lahqr vs calaman.laqr0,
 *        LAPACK's ?hseqr
 *
 * Computes the eigenvalues (@p job = Eigenvalues) or the full real Schur form
 * (@p job = Schur) of the active window @p ilo..@p ihi of an N-by-N upper
 * Hessenberg matrix @p h, optionally accumulating the orthogonal transform into
 * @p z (@p compz = Identity starts @p z at the identity, Vectors onto a caller-
 * supplied @p z, None skips it). REAL ONLY (float / double).
 *
 * This is the netlib ?hseqr DRIVER, which adds three things to the raw QR
 * kernels: (1) the eigenvalues ?gebal isolated outside @p ilo..@p ihi are copied
 * straight off @p h's diagonal; (2) @p z is initialised and the quick returns
 * (@p n <= 1) handled; (3) at the ILAENV(12) crossover NMIN = 75 it dispatches
 * -- calaman.laqr0 (small-bulge multishift QR with recursive AED) for @p n >
 * NMIN, else calaman.lahqr (double-shift Francis QR) -- and finally clears the
 * strictly-lower trash below the subdiagonal with calaman.laset. A HOST
 * COMPOSITION like calaman.laqr0: the driver logic and the O(1) scalars it
 * inspects run on the host (staged through device memcpies), the heavy work is
 * the dispatched kernel; no device code, no .cu, no bridge header.
 *
 * Allocation-free shipped surface (CLAUDE.md): @p h, @p wr, @p wi, @p z, @p work
 * and @p info are caller device pointers. @p work / @p lwork are sized exactly as
 * calaman.laqr0 requires for order @p n (see its header) -- hseqr forwards them
 * to laqr0 unchanged, and the lahqr path leaves them untouched.
 *
 * ONE DELIBERATE GAP vs the reference: the rare "?lahqr failed, retry with
 * ?laqr0" fallback is implemented only for @p n >= 49 (NL), where laqr0 runs on
 * @p h in place; the reference's @p n < 49 branch copies @p h into a 49-wide
 * scratch array first, which this allocation-free port cannot do, so there @p
 * info keeps the ?lahqr failure index rather than recovering. Unreachable for a
 * well-formed Hessenberg (?lahqr does not fail), and no QR kernel this project
 * ships is bitwise netlib anyway.
 *
 * Usage:
 *   import calaman.hseqr;
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   // d_h: n x n Hessenberg; d_z: n x n accumulator; d_wr/d_wi: length n
 *   calaman::hseqr<double>(stream, HseqrJob::Schur, HseqrCompz::Identity, n, 1, n,
 *                          d_h, ldh, d_wr, d_wi, d_z, ldz, d_work, lwork, d_info);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.hseqr;

import wwr.runtime_api;     // wwrStream_t, wwrMemcpyAsync, wwrStreamSynchronize
import wwr.wrappers.common; // real_fp
import calaman.lahqr;       // calaman::lahqr -- the n <= NMIN double-shift path
import calaman.laqr0;       // calaman::laqr0 -- the n > NMIN multishift path
import calaman.laset;       // calaman::laset + Region -- Z init and trash clear
import std;

// export import, not a plain import: hseqr RETURNS calaman::Status, so a consumer
// of `import calaman.hseqr;` must see Status's member functions, not just its
// name -- the same re-export lahqr / laqr0 / laset do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// The LAPACK JOB char ('E' / 'S'), typed: whether hseqr computes only the
// eigenvalues or the full Schur form of h -- calaman.gebal's GebalJob precedent,
// a strongly-typed stand-in for the single-char selector.
export enum class HseqrJob {
  Eigenvalues, ///< 'E': eigenvalues only; h's contents on exit are unspecified.
  Schur,       ///< 'S': the full Schur form is written into h (WANTT).
};

// The LAPACK COMPZ char ('N' / 'I' / 'V'), typed: whether and how the orthogonal
// Schur-vector matrix z is formed.
export enum class HseqrCompz {
  None,     ///< 'N': z is not referenced.
  Identity, ///< 'I': z is initialised to the identity, then the transform accumulated.
  Vectors,  ///< 'V': the transform is accumulated onto the caller's z (INITZ false).
};

/// @brief Schur factorization / eigenvalues of an upper Hessenberg matrix,
///        dispatching lahqr vs laqr0 (?hseqr)
///
/// Enqueues the QR iteration on the active window @p ilo..@p ihi (1-based) of the
/// Hessenberg @p h and synchronizes internally (the host driver inspects device
/// scalars between kernels). On return @p wr / @p wi hold the eigenvalues; when
/// @p job is Schur the window of @p h is the quasi-triangular Schur form (cleared
/// below the first subdiagonal); when @p compz is not None the orthogonal
/// transform is accumulated into rows @p ilo..@p ihi of @p z (started at the
/// identity first when Identity). The eigenvalues ?gebal isolated outside
/// @p ilo..@p ihi are copied off @p h's diagonal. @p n <= 75 (NMIN) runs through
/// calaman.lahqr, larger @p n through calaman.laqr0.
///
/// @tparam T Element type (float, double)
/// @param stream Stream the launches are enqueued on; every pointer lives on its device
/// @param job Eigenvalues (values only) or Schur (full Schur form into @p h)
/// @param compz None / Identity / Vectors -- whether and how @p z is formed
/// @param n Order of @p h (and @p z when @p compz is not None)
/// @param ilo 1-based first index of the active window (from ?gebal, else 1)
/// @param ihi 1-based last index of the active window (from ?gebal, else @p n)
/// @param h Device N-by-N upper Hessenberg matrix, leading dimension @p ldh; updated in place
/// @param ldh Leading dimension of @p h
/// @param wr Device length->= @p n real parts of the eigenvalues
/// @param wi Device length->= @p n imaginary parts of the eigenvalues
/// @param z Device N-by-N accumulator, leading dimension @p ldz; touched only when @p compz != None
/// @param ldz Leading dimension of @p z
/// @param work Device scratch; sized as calaman.laqr0 requires for order @p n (its header)
/// @param lwork Governs laqr0's NWMAX / NSMAX; pass the value laqr0 would get
/// @param info Device int; 0 on success, <0 for an illegal argument (-k), or the
///        index below which convergence failed
/// @return Success, or the runtime error a kernel launch or memcpy reported
export template<wwr::real_fp T>
Status hseqr(const wwr::wwrStream_t stream, const HseqrJob job, const HseqrCompz compz, const int n,
             const int ilo, const int ihi, T *const h, const int ldh, T *const wr, T *const wi,
             T *const z, const int ldz, T *const work, const int lwork, int *const info) {
  using std::size_t;
  const T zero{0};
  const bool wantt = (job == HseqrJob::Schur);
  const bool initz = (compz == HseqrCompz::Identity);
  const bool wantz = (compz == HseqrCompz::Identity) || (compz == HseqrCompz::Vectors);

  constexpr int kNtiny = 15; // hard subdiagonal-scratch limit shared with laqr0
  constexpr int kNmin = 75;  // ILAENV(12): the lahqr/laqr0 crossover (>= NTINY)
  constexpr int kNl = 49;    // the reference's in-place-laqr0 fallback threshold

  auto Hptr = [=](int i, int j) -> T * {
    return h + static_cast<size_t>(i - 1) + static_cast<size_t>(j - 1) * ldh;
  };
  auto write_info = [&](int v) -> Status {
    CLM_TRY(wwr::wwrMemcpyAsync(info, &v, sizeof(int), wwr::wwrMemcpyHostToDevice, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    return wwr::wwrSuccess;
  };
  auto read_info = [&](int &v) -> Status {
    CLM_TRY(wwr::wwrMemcpyAsync(&v, info, sizeof(int), wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    return wwr::wwrSuccess;
  };

  // ==== Argument validation, as the driver (not the raw kernels) owes. ====
  int ierr = 0;
  if (n < 0) {
    ierr = -4;
  } else if (ilo < 1 || ilo > std::max(1, n)) {
    ierr = -5;
  } else if (ihi < std::min(ilo, n) || ihi > n) {
    ierr = -6;
  } else if (ldh < std::max(1, n)) {
    ierr = -8;
  } else if (ldz < 1 || (wantz && ldz < std::max(1, n))) {
    ierr = -12;
  }
  if (ierr != 0) {
    return write_info(ierr);
  }
  if (n == 0) {
    return write_info(0);
  }

  // ==== Copy the eigenvalues ?gebal isolated outside [ilo, ihi] off H's
  //      diagonal: WR(i) = H(i,i), WI(i) = 0 for i in [1, ilo-1] and [ihi+1, n].
  //      Empty in the usual ilo=1, ihi=n case. ====
  auto copy_isolated = [&](int lo, int hi) -> Status {
    if (lo > hi) {
      return wwr::wwrSuccess;
    }
    for (int i = lo; i <= hi; ++i) {
      CLM_TRY(wwr::wwrMemcpyAsync(wr + (i - 1), Hptr(i, i), sizeof(T), wwr::wwrMemcpyDeviceToDevice,
                                  stream));
    }
    const std::vector<T> zeros(static_cast<size_t>(hi - lo + 1), zero);
    CLM_TRY(wwr::wwrMemcpyAsync(wi + (lo - 1), zeros.data(),
                                static_cast<size_t>(hi - lo + 1) * sizeof(T),
                                wwr::wwrMemcpyHostToDevice, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream)); // zeros must outlive the async copy
    return wwr::wwrSuccess;
  };
  CLM_TRY(copy_isolated(1, ilo - 1));
  CLM_TRY(copy_isolated(ihi + 1, n));

  // ==== Initialise Z to the identity when requested (COMPZ = 'I'). ====
  if (initz) {
    CLM_TRY(laset<T>(stream, Region::A, static_cast<size_t>(n), static_cast<size_t>(n), zero, T{1},
                     z, static_cast<size_t>(ldz)));
  }

  // ==== Quick return for a 1x1 window. ====
  if (n == 1) {
    CLM_TRY(wwr::wwrMemcpyAsync(wr, Hptr(1, 1), sizeof(T), wwr::wwrMemcpyDeviceToDevice, stream));
    CLM_TRY(wwr::wwrMemcpyAsync(wi, &zero, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    return write_info(0);
  }

  // ==== Dispatch at the crossover: laqr0 for big windows, lahqr for small. ====
  int hinfo = 0;
  const int nmin = std::max(kNtiny, kNmin);
  if (n > nmin) {
    CLM_TRY(laqr0<T>(stream, wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, ilo, ihi, z, ldz, work,
                     lwork, info));
    CLM_TRY(read_info(hinfo));
  } else {
    CLM_TRY(lahqr<T>(stream, wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, ilo, ihi, z, ldz, info));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    CLM_TRY(read_info(hinfo));
    if (hinfo > 0 && n >= kNl) {
      // Rare ?lahqr failure: ?laqr0 sometimes succeeds. Enough subdiagonal
      // scratch here to run it in place on H over [ilo, kbot].
      const int kbot = hinfo;
      CLM_TRY(laqr0<T>(stream, wantt, wantz, n, ilo, kbot, h, ldh, wr, wi, ilo, ihi, z, ldz, work,
                       lwork, info));
      CLM_TRY(read_info(hinfo));
    }
  }

  // ==== Clear the strictly-lower trash below the subdiagonal (reference's
  //      DLASET('L', N-2, N-2, 0, 0, H(3,1), LDH)). ====
  if ((wantt || hinfo != 0) && n > 2) {
    CLM_TRY(laset<T>(stream, Region::L, static_cast<size_t>(n - 2), static_cast<size_t>(n - 2),
                     zero, zero, Hptr(3, 1), static_cast<size_t>(ldh)));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
  }
  return wwr::wwrSuccess;
}

} // namespace calaman
