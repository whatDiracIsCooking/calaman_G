/**
 * @file heev.cppm
 * @brief The calaman.heev module -- eigenvalues and optional eigenvectors of a
 *        complex Hermitian matrix, LAPACK's ?heev
 *
 * calaman.syev's shape over C/Z, following ?heev's call sequence:
 * calaman.lanhe (max-abs norm) -> calaman.lascl when the norm is outside
 * [rmin, rmax] -> vendor ?hetrd -> JOBZ = N: calaman.sterf; JOBZ = V: vendor
 * ?ungtr then complex calaman.steqr (COMPZ = V) -> undo the scaling on w.
 * COMPLEX ONLY; w, e and steqr's rotations are real.
 *
 * Constraints, against the reference:
 * - The scaling multiplies the whole n-by-n A (calaman.lascl has no triangle
 *   mode), so for JOBZ = N the unreferenced triangle is scaled too.
 * - Vendor ?ungtr writes A's lda padding rows, so for JOBZ = V with lda > n it
 *   and steqr run on a packed n-by-n copy in @p work, lacpy'd back into A (as
 *   real 2n-row matrices: calaman.lacpy is real-only).
 * - SYNCHRONIZES once to read the norm, and again only when scaling, to read
 *   INFO; otherwise the tail is enqueued on the solver's stream.
 * - Allocation-free apart from calaman.lanhe's own scratch.
 * Usage:
 *   import calaman.heev;
 *   using Z = wwr::wwrDoubleComplex;
 *   const std::size_t bytes = calaman::heev_bufferSize<Z>(solver, Jobz::V, Uplo::L, n, lda);
 *   calaman::heev<Z>(solver, Jobz::V, Uplo::L, n, d_a, lda, d_w, d_work, bytes, d_info);
 */

module;

#include "error_handling/error_macros.h"

export module calaman.heev;

import std;
import wwr.blas;            // WWRBLAS_FILL_MODE_UPPER / _LOWER
import wwr.solver;          // wwrsolverDnHandle_t, wwrsolverDnGetStream, status codes
import wwr.runtime_api;     // wwrStream_t, wwrMemcpyAsync / MemsetAsync, synchronize
import wwr.complex;         // wwrFloatComplex, wwrDoubleComplex
import wwr.wrappers.solver; // hetrd / ungtr + their bufferSize queries
import calaman.common;      // Jobz, Uplo, MatrixNorm, WorkspaceLayout, carve_workspace
import calaman.lanhe;       // max-abs norm for the scaling decision
import calaman.lascl;       // scale A, rescale w
import calaman.lacpy;       // A <-> the packed eigenvector copy when lda > n
import calaman.sterf;       // eigenvalues of the tridiagonal (JOBZ = N)
import calaman.steqr;       // eigenpairs of the tridiagonal (JOBZ = V)

export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

export using calaman::Jobz;
export using calaman::Uplo;

namespace heev_detail {

inline wwr::wwrblasFillMode_t fill_mode(const Uplo uplo) {
  return uplo == Uplo::U ? wwr::WWRBLAS_FILL_MODE_UPPER : wwr::WWRBLAS_FILL_MODE_LOWER;
}

/// @brief heev's device workspace: the regions live across phases, then one
///        scratch zone shared by ?hetrd, ?ungtr and steqr (never live together)
template<typename T>
struct HeevSlices {
  using R = ComplexToRealType<T>;
  R *anrm = nullptr;    ///< lanhe's max-abs result
  R *e = nullptr;       ///< tridiagonal off-diagonal (n), real
  T *tau = nullptr;     ///< ?hetrd reflector scalars (n), read by ?ungtr
  T *z = nullptr;       ///< packed n-by-n eigenvectors; JOBZ = V with lda > n only
  int *vinfo = nullptr; ///< vendor devInfo, kept apart from the caller's INFO
  T *hetrd_work = nullptr;
  T *ungtr_work = nullptr;
  void *steqr_work = nullptr; ///< steqr's real rotation region (?heev's RWORK)
  int hetrd_lwork = 0;
  int ungtr_lwork = 0;
  std::size_t steqr_bytes = 0;

  void carve(WorkspaceLayout &layout, const bool wantz, const bool packed, const int n,
             const int hetrd_len, const int ungtr_len, const std::size_t steqr_len) {
    const std::size_t nsz = static_cast<std::size_t>(n < 1 ? 1 : n);
    anrm = layout.fixed<R>(1);
    e = layout.fixed<R>(nsz);
    tau = layout.fixed<T>(nsz);
    vinfo = layout.fixed<int>(1);
    if (wantz && packed) {
      z = layout.fixed<T>(nsz * nsz);
    }
    hetrd_work = layout.scratch<T>(static_cast<std::size_t>(hetrd_len));
    hetrd_lwork = hetrd_len;
    if (wantz) {
      ungtr_work = layout.scratch<T>(static_cast<std::size_t>(ungtr_len));
      ungtr_lwork = ungtr_len;
      steqr_work = layout.scratch<std::byte>(steqr_len);
      steqr_bytes = steqr_len;
    }
  }
};

static_assert(
    slices_for<HeevSlices<wwr::wwrDoubleComplex>, bool, bool, int, int, int, std::size_t>);

/// @brief Query the vendor lwork lengths, then size (@p base null) or carve
///        the workspace; returns the bytes the layout spans
template<typename T>
std::size_t map_workspace(wwr::wwrsolverDnHandle_t solver, void *base, const Jobz jobz,
                          const Uplo uplo, const int n, const int lda, HeevSlices<T> *out) {
  using R = ComplexToRealType<T>;
  const bool wantz = jobz == Jobz::V;
  const bool packed = lda > n;
  const int ldz = packed ? std::max(1, n) : lda;
  int hetrd_len = 1;
  int ungtr_len = 1;
  if (n > 0) {
    const T *dummy = nullptr;
    const R *rdummy = nullptr;
    int lw = 0;
    if (wwr::hetrd_bufferSize<T>(solver, fill_mode(uplo), n, dummy, lda, rdummy, rdummy, dummy,
                                 &lw) == wwr::WWRSOLVER_STATUS_SUCCESS) {
      hetrd_len = std::max(1, lw);
    }
    lw = 0;
    if (wantz && wwr::ungtr_bufferSize<T>(solver, fill_mode(uplo), n, dummy, ldz, dummy, &lw) ==
                     wwr::WWRSOLVER_STATUS_SUCCESS) {
      ungtr_len = std::max(1, lw);
    }
  }
  const std::size_t steqr_len = wantz ? steqr_bufferSize<T>(CompZ::V, n) : std::size_t{0};
  return carve_workspace(base, out, wantz, packed, n, hetrd_len, ungtr_len, steqr_len);
}

/// @brief Copy the n-by-n complex @p src (ld @p lds) to @p dst (ld @p ldd) as
///        the real 2n-by-n matrix of its interleaved parts
template<typename T>
Status copy_square(const wwr::wwrStream_t stream, const std::size_t n, const T *src,
                   const std::size_t lds, T *dst, const std::size_t ldd) {
  using R = ComplexToRealType<T>;
  return lacpy<R>(stream, Region::A, 2 * n, n, reinterpret_cast<const R *>(src), 2 * lds,
                  reinterpret_cast<R *>(dst), 2 * ldd);
}

} // namespace heev_detail

/// @brief Device workspace heev() needs, in bytes
///
/// Covers the norm scalar, e, tau, the vendor devInfo, the packed n-by-n
/// eigenvectors (Jobz::V, lda > n) and the scratch ?hetrd / ?ungtr / steqr
/// share. Pass the same @p solver and @p lda to heev().
export template<calaman::complex_fp T>
std::size_t heev_bufferSize(wwr::wwrsolverDnHandle_t solver, const Jobz jobz, const Uplo uplo,
                            const int n, const int lda) {
  return heev_detail::map_workspace<T>(solver, nullptr, jobz, uplo, n, lda, nullptr);
}

/// @brief Eigenvalues and optional eigenvectors of a complex Hermitian matrix (?heev)
///
/// On return @p w holds the eigenvalues in ascending order; for Jobz::V @p a
/// holds the orthonormal eigenvectors, else its @p uplo triangle is destroyed.
/// @p info (device int) gets 0, or ?sterf / ?steqr's count of unconverged
/// off-diagonals.
///
/// @tparam T Element type (wwrFloatComplex, wwrDoubleComplex)
/// @param solver Solver handle; its stream carries the whole run
/// @param a Device n-by-n matrix, column-major, leading dim @p lda (>= max(1, n))
/// @param w Device length-n real eigenvalues
/// @param work_bytes Size of @p work; at least heev_bufferSize<T>(solver, jobz, uplo, n, lda)
/// @return Success, invalid-value for a bad argument, or the failing step's error
export template<calaman::complex_fp T>
Status heev(wwr::wwrsolverDnHandle_t solver, const Jobz jobz, const Uplo uplo, const int n,
            T *const a, const int lda, ComplexToRealType<T> *const w, void *const work,
            const std::size_t work_bytes, int *const info) {
  using std::size_t;
  using R = ComplexToRealType<T>;
  const bool wantz = jobz == Jobz::V;
  CLM_REQUIRE(n >= 0 && lda >= std::max(1, n), wwr::wwrErrorInvalidValue);
  CLM_REQUIRE(info != nullptr, wwr::wwrErrorInvalidValue);
  CLM_REQUIRE(n == 0 || (a != nullptr && w != nullptr), wwr::wwrErrorInvalidValue);

  wwr::wwrStream_t stream{};
  CLM_REQUIRE(wwr::wwrsolverDnGetStream(solver, &stream) == wwr::WWRSOLVER_STATUS_SUCCESS,
              wwr::wwrErrorInvalidValue);

  if (n == 0) {
    return wwr::wwrMemsetAsync(info, 0, sizeof(int), stream);
  }
  if (n == 1) { // ?heev's quick return: w = real(a(1,1)), Z = 1
    CLM_TRY(wwr::wwrMemcpyAsync(w, a, sizeof(R), wwr::wwrMemcpyDeviceToDevice, stream));
    if (wantz) {
      const R one[2] = {R{1}, R{0}};
      CLM_TRY(wwr::wwrMemcpyAsync(a, one, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
      CLM_TRY(wwr::wwrStreamSynchronize(stream)); // `one` is a host stack value
    }
    return wwr::wwrMemsetAsync(info, 0, sizeof(int), stream);
  }

  heev_detail::HeevSlices<T> ws;
  const size_t need = heev_detail::map_workspace<T>(solver, work, jobz, uplo, n, lda, &ws);
  CLM_REQUIRE(work != nullptr && work_bytes >= need, wwr::wwrErrorInvalidValue);

  // ==== Scale A to [rmin, rmax] when its max-abs norm is out of range. ====
  const R eps = std::numeric_limits<R>::epsilon(); // DLAMCH 'P'
  const R smlnum = std::numeric_limits<R>::min() / eps;
  const R rmin = std::sqrt(smlnum);
  const R rmax = std::sqrt(R{1} / smlnum);
  const auto un = static_cast<size_t>(n);
  const auto ulda = static_cast<size_t>(lda);
  CLM_TRY(lanhe<T>(stream, MatrixNorm::max_abs, uplo, un, a, ulda, ws.anrm));
  R anrm{};
  CLM_TRY(wwr::wwrMemcpyAsync(&anrm, ws.anrm, sizeof(R), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  R sigma{1};
  const bool iscale = (anrm > R{0} && anrm < rmin) || anrm > rmax;
  if (iscale) {
    sigma = (anrm < rmin ? rmin : rmax) / anrm;
    CLM_TRY(lascl<T>(stream, R{1}, sigma, un, un, a, ulda));
  }

  // ==== Tridiagonal reduction (vendor ?hetrd): d into w, then the eigensolve. ====
  const auto fill = heev_detail::fill_mode(uplo);
  {
    const auto st = wwr::hetrd<T>(solver, fill, n, a, lda, w, ws.e, ws.tau, ws.hetrd_work,
                                  ws.hetrd_lwork, ws.vinfo);
    if (st != wwr::WWRSOLVER_STATUS_SUCCESS) {
      return st;
    }
  }
  if (!wantz) {
    CLM_TRY(sterf<R>(stream, n, w, ws.e, info));
  } else {
    T *const z = ws.z != nullptr ? ws.z : a;
    const int ldz = ws.z != nullptr ? n : lda;
    if (z != a) {
      CLM_TRY(heev_detail::copy_square<T>(stream, un, a, ulda, z, un));
    }
    const auto st =
        wwr::ungtr<T>(solver, fill, n, z, ldz, ws.tau, ws.ungtr_work, ws.ungtr_lwork, ws.vinfo);
    if (st != wwr::WWRSOLVER_STATUS_SUCCESS) {
      return st;
    }
    CLM_TRY(steqr<T>(stream, CompZ::V, n, w, ws.e, z, ldz, ws.steqr_work, ws.steqr_bytes, info));
    if (z != a) {
      CLM_TRY(heev_detail::copy_square<T>(stream, un, z, un, a, ulda));
    }
  }

  // ==== Undo the scaling on the converged eigenvalues (IMAX of ?heev). ====
  if (iscale) {
    int hinfo = 0;
    CLM_TRY(wwr::wwrMemcpyAsync(&hinfo, info, sizeof(int), wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    const int imax = hinfo == 0 ? n : hinfo - 1;
    if (imax > 0) {
      CLM_TRY(lascl<R>(stream, sigma, R{1}, static_cast<size_t>(imax), 1, w,
                       static_cast<size_t>(imax)));
    }
  }
  return wwr::wwrSuccess;
}

} // namespace calaman
