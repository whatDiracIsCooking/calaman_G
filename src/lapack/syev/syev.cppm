/**
 * @file syev.cppm
 * @brief The calaman.syev module -- eigenvalues and optional eigenvectors of a
 *        real symmetric matrix, LAPACK's ?syev
 *
 * A host composition in calaman.geev's shape, following ?syev's call sequence:
 * calaman.lansy (max-abs norm) -> calaman.lascl when the norm is outside
 * [rmin, rmax] -> vendor ?sytrd -> JOBZ = N: calaman.sterf; JOBZ = V: vendor
 * ?orgtr then calaman.steqr (COMPZ = V) -> undo the scaling on w. REAL ONLY.
 *
 * Constraints, against the reference:
 * - The scaling multiplies the whole n-by-n A (calaman.lascl has no triangle
 *   mode), so for JOBZ = N the unreferenced triangle is scaled too.
 * - The rescale of w is calaman.lascl(sigma -> 1): one multiply by 1/sigma,
 *   the ?scal the reference does, without a BLAS handle.
 * - SYNCHRONIZES once to read the norm, and again only when scaling, to read
 *   INFO; otherwise the tail is enqueued on the solver's stream.
 * - Allocation-free apart from calaman.lansy's own scratch; @p work is one
 *   buffer of syev_bufferSize<T>() bytes.
 *
 * Usage:
 *   import calaman.syev;
 *   const std::size_t bytes = calaman::syev_bufferSize<double>(solver, Jobz::V, Uplo::L, n, lda);
 *   calaman::syev<double>(solver, Jobz::V, Uplo::L, n, d_a, lda, d_w, d_work, bytes, d_info);
 */

module;

#include "error_handling/error_macros.h"

export module calaman.syev;

import std;
import wwr.blas;            // WWRBLAS_FILL_MODE_UPPER / _LOWER
import wwr.solver;          // wwrsolverDnHandle_t, wwrsolverDnGetStream, status codes
import wwr.runtime_api;     // wwrStream_t, wwrMemcpyAsync / MemsetAsync, synchronize
import wwr.wrappers.solver; // sytrd / orgtr + their bufferSize queries
import calaman.common;      // Jobz, Uplo, MatrixNorm, WorkspaceLayout, carve_workspace
import calaman.lansy;       // max-abs norm for the scaling decision
import calaman.lascl;       // scale A, rescale w
import calaman.sterf;       // eigenvalues of the tridiagonal (JOBZ = N)
import calaman.steqr;       // eigenpairs of the tridiagonal (JOBZ = V)

export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

export using calaman::Jobz;
export using calaman::Uplo;

namespace syev_detail {

inline wwr::wwrblasFillMode_t fill_mode(const Uplo uplo) {
  return uplo == Uplo::U ? wwr::WWRBLAS_FILL_MODE_UPPER : wwr::WWRBLAS_FILL_MODE_LOWER;
}

/// @brief syev's device workspace: the regions live across phases, then one
///        scratch zone shared by ?sytrd, ?orgtr and steqr (never live together)
template<typename T>
struct SyevSlices {
  T *anrm = nullptr;      ///< lansy's max-abs result
  T *e = nullptr;         ///< tridiagonal off-diagonal (n)
  T *tau = nullptr;       ///< ?sytrd reflector scalars (n), read by ?orgtr
  int *vinfo = nullptr;   ///< vendor devInfo, kept apart from the caller's INFO
  T *sytrd_work = nullptr;
  T *orgtr_work = nullptr;
  void *steqr_work = nullptr;
  int sytrd_lwork = 0;
  int orgtr_lwork = 0;
  std::size_t steqr_bytes = 0;

  void carve(WorkspaceLayout &layout, const bool wantz, const int n, const int sytrd_len,
             const int orgtr_len, const std::size_t steqr_len) {
    const std::size_t nsz = static_cast<std::size_t>(n < 1 ? 1 : n);
    anrm = layout.fixed<T>(1);
    e = layout.fixed<T>(nsz);
    tau = layout.fixed<T>(nsz);
    vinfo = layout.fixed<int>(1);
    sytrd_work = layout.scratch<T>(static_cast<std::size_t>(sytrd_len));
    sytrd_lwork = sytrd_len;
    if (wantz) {
      orgtr_work = layout.scratch<T>(static_cast<std::size_t>(orgtr_len));
      orgtr_lwork = orgtr_len;
      steqr_work = layout.scratch<std::byte>(steqr_len);
      steqr_bytes = steqr_len;
    }
  }
};

static_assert(slices_for<SyevSlices<double>, bool, int, int, int, std::size_t>);

/// @brief Query the vendor lwork lengths, then size (@p base null) or carve
///        the workspace; returns the bytes the layout spans
template<typename T>
std::size_t map_workspace(wwr::wwrsolverDnHandle_t solver, void *base, const Jobz jobz,
                          const Uplo uplo, const int n, const int lda, SyevSlices<T> *out) {
  const bool wantz = jobz == Jobz::V;
  int sytrd_len = 1;
  int orgtr_len = 1;
  if (n > 0) {
    const T *dummy = nullptr;
    int lw = 0;
    if (wwr::sytrd_bufferSize<T>(solver, fill_mode(uplo), n, dummy, lda, dummy, dummy, dummy,
                                 &lw) == wwr::WWRSOLVER_STATUS_SUCCESS) {
      sytrd_len = std::max(1, lw);
    }
    lw = 0;
    if (wantz && wwr::orgtr_bufferSize<T>(solver, fill_mode(uplo), n, dummy, lda, dummy, &lw) ==
                     wwr::WWRSOLVER_STATUS_SUCCESS) {
      orgtr_len = std::max(1, lw);
    }
  }
  const std::size_t steqr_len = wantz ? steqr_bufferSize<T>(CompZ::V, n) : std::size_t{0};
  return carve_workspace(base, out, wantz, n, sytrd_len, orgtr_len, steqr_len);
}

} // namespace syev_detail

/// @brief Device workspace syev() needs, in bytes
///
/// Covers the norm scalar, e, tau, the vendor devInfo and the scratch zone
/// ?sytrd / ?orgtr / steqr share. Pass the same @p solver and @p lda to syev().
export template<calaman::real_fp T>
std::size_t syev_bufferSize(wwr::wwrsolverDnHandle_t solver, const Jobz jobz, const Uplo uplo,
                            const int n, const int lda) {
  return syev_detail::map_workspace<T>(solver, nullptr, jobz, uplo, n, lda, nullptr);
}

/// @brief Eigenvalues and optional eigenvectors of a real symmetric matrix (?syev)
///
/// On return @p w holds the eigenvalues in ascending order; for Jobz::V @p a
/// holds the orthonormal eigenvectors, else its @p uplo triangle is destroyed.
/// @p info (device int) gets 0, or ?sterf / ?steqr's count of unconverged
/// off-diagonals.
///
/// @tparam T Element type (float, double)
/// @param solver Solver handle; its stream carries the whole run
/// @param a Device n-by-n matrix, column-major, leading dim @p lda (>= max(1, n))
/// @param w Device length-n eigenvalues
/// @param work_bytes Size of @p work; at least syev_bufferSize<T>(solver, jobz, uplo, n, lda)
/// @return Success, invalid-value for a bad argument, or the failing step's error
export template<calaman::real_fp T>
Status syev(wwr::wwrsolverDnHandle_t solver, const Jobz jobz, const Uplo uplo, const int n,
            T *const a, const int lda, T *const w, void *const work, const std::size_t work_bytes,
            int *const info) {
  using std::size_t;
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
  if (n == 1) { // ?syev's quick return: w = a(1,1), Z = 1
    CLM_TRY(wwr::wwrMemcpyAsync(w, a, sizeof(T), wwr::wwrMemcpyDeviceToDevice, stream));
    if (wantz) {
      const T one{1};
      CLM_TRY(wwr::wwrMemcpyAsync(a, &one, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
      CLM_TRY(wwr::wwrStreamSynchronize(stream)); // `one` is a host stack value
    }
    return wwr::wwrMemsetAsync(info, 0, sizeof(int), stream);
  }

  syev_detail::SyevSlices<T> ws;
  const size_t need = syev_detail::map_workspace<T>(solver, work, jobz, uplo, n, lda, &ws);
  CLM_REQUIRE(work != nullptr && work_bytes >= need, wwr::wwrErrorInvalidValue);

  // ==== Scale A to [rmin, rmax] when its max-abs norm is out of range. ====
  const T eps = std::numeric_limits<T>::epsilon(); // DLAMCH 'P'
  const T smlnum = std::numeric_limits<T>::min() / eps;
  const T rmin = std::sqrt(smlnum);
  const T rmax = std::sqrt(T{1} / smlnum);
  const auto un = static_cast<size_t>(n);
  const auto ulda = static_cast<size_t>(lda);
  CLM_TRY(lansy<T>(stream, MatrixNorm::max_abs, uplo, un, a, ulda, ws.anrm));
  T anrm{};
  CLM_TRY(wwr::wwrMemcpyAsync(&anrm, ws.anrm, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  T sigma{1};
  const bool iscale = (anrm > T{0} && anrm < rmin) || anrm > rmax;
  if (iscale) {
    sigma = (anrm < rmin ? rmin : rmax) / anrm;
    CLM_TRY(lascl<T>(stream, T{1}, sigma, un, un, a, ulda));
  }

  // ==== Tridiagonal reduction (vendor ?sytrd): d into w, then the eigensolve. ====
  const auto fill = syev_detail::fill_mode(uplo);
  {
    const auto st = wwr::sytrd<T>(solver, fill, n, a, lda, w, ws.e, ws.tau, ws.sytrd_work,
                                  ws.sytrd_lwork, ws.vinfo);
    if (st != wwr::WWRSOLVER_STATUS_SUCCESS) {
      return st;
    }
  }
  if (!wantz) {
    CLM_TRY(sterf<T>(stream, n, w, ws.e, info));
  } else {
    const auto st =
        wwr::orgtr<T>(solver, fill, n, a, lda, ws.tau, ws.orgtr_work, ws.orgtr_lwork, ws.vinfo);
    if (st != wwr::WWRSOLVER_STATUS_SUCCESS) {
      return st;
    }
    CLM_TRY(steqr<T>(stream, CompZ::V, n, w, ws.e, a, lda, ws.steqr_work, ws.steqr_bytes, info));
  }

  // ==== Undo the scaling on the converged eigenvalues (IMAX of ?syev). ====
  if (iscale) {
    int hinfo = 0;
    CLM_TRY(wwr::wwrMemcpyAsync(&hinfo, info, sizeof(int), wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    const int imax = hinfo == 0 ? n : hinfo - 1;
    if (imax > 0) {
      CLM_TRY(lascl<T>(stream, sigma, T{1}, static_cast<size_t>(imax), 1, w,
                       static_cast<size_t>(imax)));
    }
  }
  return wwr::wwrSuccess;
}

} // namespace calaman
