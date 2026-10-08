/**
 * @file shift_invert.cppm
 * @brief The calaman.shift_invert module -- DenseShiftInvert, the
 *        linear_operator of (A - sigma I)^{-1} for a dense real symmetric A
 *
 * prepare copies the @p uplo triangle of A into the workspace, subtracts sigma
 * on its diagonal and factors it once by Bunch-Kaufman (wwr::sytrf); apply then
 * solves against that factor (calaman.sytrs2). A shift-invert eigensolver hands
 * the model to its driver and back-transforms with sigma(): lambda = sigma + 1/theta.
 *
 * Its own module, not a solver partition, so any linear_operator consumer
 * (lanczos, davidson) can take it. Header-only templates; the device work
 * arrives through calaman.lacpy, calaman.laset and calaman.sytrs2.
 *
 * Every member expects both handles to be bound to the @p stream it is given.
 *
 * Usage:
 *   import calaman.shift_invert;
 *
 *   std::size_t bytes = 0;
 *   calaman::dense_shift_invert_bufferSize<double>(solver, n, &bytes);
 *   calaman::DenseShiftInvertSlices<double> s;
 *   calaman::make_dense_shift_invert_slices<double>(solver, n, d_work, &s, nullptr);
 *   calaman::DenseShiftInvert<double> op{blas, solver, calaman::Uplo::L, n, d_A, lda, sigma, s};
 *   op.prepare(stream);           // singular shift: a failing Status
 *   op.apply(stream, k, d_X, d_Y); // Y = (A - sigma I)^{-1} X
 */

module;

// CLM_TRY / CLM_REQUIRE -- macros, so they arrive by #include in the global
// module fragment, not by import. Resolved root-relative via the src/ root
// calaman.error_handling exports; need calaman::Status visible at expansion,
// which the import below (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.shift_invert;

import std;
import wwr.runtime_api;     // wwrStream_t, wwrMemcpyAsync, wwrStreamSynchronize
import wwr.blas;            // wwrblasHandle_t, WWRBLAS_*
import wwr.solver;          // wwrsolverDnHandle_t
import wwr.wrappers.blas;   // axpy
import wwr.wrappers.solver; // sytrf, sytrf_bufferSize
import wwr.extension.blas;  // ScopedPointerMode
import calaman.common;      // real_fp, Uplo, Region, kOne, carve_workspace, WorkspaceLayout
import calaman.lacpy;       // lacpy -- A's triangle into the factor region
import calaman.laset;       // laset -- the -sigma vector the diagonal shift adds
import calaman.sytrs2;      // sytrs2 -- the solve against the Bunch-Kaufman factor
export import calaman.linear_operator; // linear_operator
export import calaman.error_handling;  // Status, PointerModeStatus, devinfo_verdict

export namespace calaman {

/// @brief DenseShiftInvert's workspace: the factor, its pivots and info are
///        FIXED; the shift vector, the sytrf workspace and sytrs2's E vector are
///        never live together, so they share one scratch zone.
template<calaman::real_fp T>
struct DenseShiftInvertSlices {
  T *factor = nullptr;     ///< n x n, ld n: A - sigma I, then its Bunch-Kaufman factor
  int *ipiv = nullptr;     ///< n sytrf pivots
  int *info = nullptr;     ///< the sytrf devInfo
  T *shift = nullptr;      ///< n copies of -sigma (prepare, before sytrf)
  T *sytrf_work = nullptr; ///< lwork_sytrf elements (prepare)
  T *e = nullptr;          ///< n: sytrs2's E vector (apply)
  int n = 0;               ///< order the layout was sized for
  int lwork_sytrf = 0;     ///< elements of sytrf_work

  /// @brief Lay the slices out from @p layout for order @p order and a vendor
  ///        sytrf workspace of @p sytrf_len elements.
  void carve(WorkspaceLayout &layout, const int order, const int sytrf_len) {
    const std::size_t nz = static_cast<std::size_t>(order);
    n = order;
    lwork_sytrf = sytrf_len;
    factor = layout.fixed<T>(nz * nz);
    ipiv = layout.fixed<int>(nz);
    info = layout.fixed<int>(1);
    shift = layout.scratch<T>(nz);
    sytrf_work = layout.scratch<T>(static_cast<std::size_t>(sytrf_len));
    e = layout.scratch<T>(nz);
  }
};

/**
 * @brief Query sytrf's workspace and carve @p d_work into DenseShiftInvertSlices
 *        for an n x n A.
 *
 * @param d_work Workspace, or null to size it only. @p slices, @p lwork_bytes: out, may be null.
 * @return INVALID_VALUE for n < 1, or the solver-domain status of a failed query.
 */
template<calaman::real_fp T>
Status make_dense_shift_invert_slices(wwr::wwrsolverDnHandle_t solver_handle, const int n,
                                      void *d_work, DenseShiftInvertSlices<T> *slices,
                                      std::size_t *lwork_bytes) {
  CLM_REQUIRE(n >= 1, wwr::WWRBLAS_STATUS_INVALID_VALUE);
  int lwork = 0;
  CLM_TRY(wwr::sytrf_bufferSize<T>(solver_handle, n, static_cast<T *>(nullptr), n, &lwork));
  const std::size_t bytes = carve_workspace(d_work, slices, n, std::max(lwork, 1));
  if (lwork_bytes != nullptr) {
    *lwork_bytes = bytes;
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief Device workspace, in bytes, that DenseShiftInvert needs at order @p n.
/// @return INVALID_VALUE for a null @p lwork_bytes or n < 1.
template<calaman::real_fp T>
Status dense_shift_invert_bufferSize(wwr::wwrsolverDnHandle_t solver_handle, const int n,
                                     std::size_t *lwork_bytes) {
  CLM_REQUIRE(lwork_bytes != nullptr, wwr::WWRBLAS_STATUS_INVALID_VALUE);
  return make_dense_shift_invert_slices<T>(solver_handle, n, nullptr, nullptr, lwork_bytes);
}

/**
 * @brief The linear_operator Y = (A - sigma I)^{-1} X of a dense real symmetric
 *        A, of which only the @p uplo triangle is read. Holds A and its carved
 *        workspace, owning neither; A is read only by prepare.
 */
template<calaman::real_fp T>
class DenseShiftInvert {
public:
  /// @param s Carved by make_dense_shift_invert_slices for this @p n.
  DenseShiftInvert(wwr::wwrblasHandle_t blas_handle, wwr::wwrsolverDnHandle_t solver_handle,
                   const Uplo uplo, const int n, const T *d_A, const int lda, const T sigma,
                   const DenseShiftInvertSlices<T> &s)
      : blas_{blas_handle}, solver_{solver_handle}, uplo_{uplo}, n_{n}, d_A_{d_A}, lda_{lda},
        sigma_{sigma}, s_{s} {}

  /// @brief Form A - sigma I and factor it. Synchronizes @p stream once, to read
  ///        sytrf's info: a singular shift comes back as devinfo_verdict's failure.
  /// @return INVALID_VALUE for a shape the slices were not carved for.
  Status prepare(wwr::wwrStream_t stream) {
    factored_ = false;
    CLM_REQUIRE(n_ >= 1 && n_ == s_.n && lda_ >= n_ && d_A_ != nullptr,
                wwr::WWRBLAS_STATUS_INVALID_VALUE);
    const auto nz = static_cast<std::size_t>(n_);
    const Region region = uplo_ == Uplo::U ? Region::U : Region::L;

    CLM_TRY(lacpy<T>(stream, region, nz, nz, d_A_, static_cast<std::size_t>(lda_), s_.factor, nz));
    CLM_TRY(laset<T>(stream, Region::A, nz, 1, -sigma_, -sigma_, s_.shift, nz));
    {
      wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
      const wwr::extension::ScopedPointerMode mode{blas_, wwr::WWRBLAS_POINTER_MODE_HOST,
                                                   PointerModeStatus{&pm_status}};
      CLM_TRY(pm_status);
      CLM_TRY(wwr::axpy<T, int>(blas_, n_, &kOne<T>, s_.shift, 1, s_.factor, n_ + 1));
    }

    const wwr::wwrblasFillMode_t fill =
        uplo_ == Uplo::U ? wwr::WWRBLAS_FILL_MODE_UPPER : wwr::WWRBLAS_FILL_MODE_LOWER;
    CLM_TRY(wwr::sytrf<T>(solver_, fill, n_, s_.factor, n_, s_.ipiv, s_.sytrf_work, s_.lwork_sytrf,
                          s_.info));
    int info = 0;
    CLM_TRY(wwr::wwrMemcpyAsync(&info, s_.info, sizeof(int), wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    CLM_TRY(devinfo_verdict(info));
    factored_ = true;
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  /// @brief Y = (A - sigma I)^{-1} X, X and Y n x k, ld n, unaliased.
  /// @return INVALID_VALUE before a successful prepare, or for k < 0.
  Status apply(wwr::wwrStream_t stream, const int k, const T *X, T *Y) {
    CLM_REQUIRE(factored_ && k >= 0, wwr::WWRBLAS_STATUS_INVALID_VALUE);
    if (k == 0) {
      return wwr::WWRBLAS_STATUS_SUCCESS;
    }
    const std::size_t nk = static_cast<std::size_t>(n_) * static_cast<std::size_t>(k);
    CLM_TRY(wwr::wwrMemcpyAsync(Y, X, nk * sizeof(T), wwr::wwrMemcpyDeviceToDevice, stream));

    wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
    const wwr::extension::ScopedPointerMode mode{blas_, wwr::WWRBLAS_POINTER_MODE_HOST,
                                                 PointerModeStatus{&pm_status}};
    CLM_TRY(pm_status);
    CLM_TRY(sytrs2<T>(blas_, uplo_, n_, k, s_.factor, n_, s_.ipiv, Y, n_, s_.e));
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  /// @brief The shift, for the back-transform lambda = sigma + 1/theta.
  T sigma() const noexcept { return sigma_; }
  /// @brief The dimension n.
  int dim() const noexcept { return n_; }

private:
  wwr::wwrblasHandle_t blas_;
  wwr::wwrsolverDnHandle_t solver_;
  Uplo uplo_;
  int n_;
  const T *d_A_;
  int lda_;
  T sigma_;
  DenseShiftInvertSlices<T> s_;
  bool factored_ = false;
};

} // namespace calaman

namespace calaman {

static_assert(slices_for<DenseShiftInvertSlices<float>, int, int>);
static_assert(linear_operator<DenseShiftInvert<float>, float>);
static_assert(linear_operator<DenseShiftInvert<double>, double>);
static_assert(!linear_operator<DenseShiftInvert<double>, float>);

} // namespace calaman
