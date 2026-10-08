/**
 * @file resolvent.cppm
 * @brief feast_resolvent, the model the FEAST driver filters through, and
 *        DenseResolvent, its model for a dense real symmetric A
 *
 * The :resolvent partition of calaman.feast.
 *
 *   filter(Y) = rho(A) Y = sum_e Re[ w_e (Z_e I - A)^{-1} Y ]   (:compute_quadrature)
 *
 * A model is a linear_operator (calaman.linear_operator) that can also apply the
 * filter: prepare once per solve, filter once per iteration. It owns its
 * workspace and its failure reporting; the driver sees only Status.
 *
 * DenseResolvent factors all Ne resolvents in one getrfBatched per solve and
 * solves them in one getrsBatched per iteration; the Ne shifts are never split
 * into separate calls. Z_e I - A is complex symmetric, so the factorization is
 * a general LU, never singular: every Im Z_e > 0 and A's spectrum is real.
 *
 * A model may add the optional feast_norm1_hook, ||A||_1 for the residuals'
 * scale; DenseResolvent's is exact. An inexact model may add feast_residual_hook,
 * the residual form; DenseResolvent's solves are exact, so it keeps the plain one.
 *
 * Every member expects the BLAS handle's stream to be the @p stream it is
 * given.
 */

module;

#include "feast_bridge.h"

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.feast:resolvent;

import std;
import wwr.runtime_api;    // wwrStream_t, wwrGetLastError, wwrMemcpyAsync, wwrStreamSynchronize
import wwr.blas;           // wwrblasHandle_t, WWRBLAS_*, wwrblasFillMode_t
import wwr.wrappers.blas;  // getrfBatched, getrsBatched, symm
import wwr.extension.blas; // ScopedPointerMode
import calaman.common;     // real_fp, RealToComplexType, kOne, kZero, align_up, WorkspaceLayout
export import calaman.linear_operator; // linear_operator
export import calaman.error_handling;  // Status, PointerModeStatus, devinfo_verdict

export namespace calaman {

/// @brief A linear_operator that also applies the FEAST filter for a contour.
///
/// r.prepare(stream, contour) once per solve, then r.filter(stream, contour, k, Y, out):
/// out = Re sum_e w_e (Z_e I - A)^{-1} Y, Y and out n x k on the device, ld n, unaliased.
/// r.dim() is n; r.k_max() and r.shifts_max() are the widest k and most nodes it takes.
template<class R, class T>
concept feast_resolvent = linear_operator<R, T> && requires(R &r, wwr::wwrStream_t stream,
                                                            const device::FeastContour<T> &contour,
                                                            int k, const T *Y, T *out) {
  { r.prepare(stream, contour) } -> std::convertible_to<Status>;
  { r.filter(stream, contour, k, Y, out) } -> std::convertible_to<Status>;
  { r.dim() } -> std::convertible_to<int>;
  { r.k_max() } -> std::convertible_to<int>;
  { r.shifts_max() } -> std::convertible_to<int>;
};

/// @brief The optional hook: r.norm1_estimate(stream, d_out) enqueues ||A||_1,
///        or a lower bound on it, into the device scalar @p d_out. A model
///        without it gets lacn2's estimate, driven by its apply.
template<class R, class T>
concept feast_norm1_hook = requires(R &r, wwr::wwrStream_t stream, T *d_out) {
  { r.norm1_estimate(stream, d_out) } -> std::convertible_to<Status>;
};

/// @brief The optional residual-form hook (IFEAST): r.filter_residual(stream,
///        contour, k, X, lambda, R, out) enqueues out = rho(A) X for Ritz pairs
///        (lambda_j, x_j) given R = A X - X diag(lambda), so the inner solves'
///        right-hand side is R. A model without it is filtered by filter alone.
template<class R, class T>
concept feast_residual_hook =
    requires(R &r, wwr::wwrStream_t stream, const device::FeastContour<T> &contour, int k,
             const T *X, const T *lambda, const T *res, T *out) {
      { r.filter_residual(stream, contour, k, X, lambda, res, out) } -> std::convertible_to<Status>;
    };

} // namespace calaman

namespace calaman {

/// @brief DenseResolvent's workspace: Ne resolvents and right-hand-side blocks,
///        their batched pointer arrays, pivots, per-node LU info, and ||A||_1 scratch.
template<calaman::real_fp T>
struct DenseResolventSlices {
  using C = calaman::RealToComplexType<T>;

  C *resolvents = nullptr;          ///< Ne packed n x n blocks: Z_e I - A, then its LU factors
  C *rhs = nullptr;                 ///< Ne n x k_max blocks: Y, then (Z_e I - A)^{-1} Y
  std::size_t resolvent_stride = 0; ///< elements from one resolvent block to the next
  std::size_t rhs_stride = 0;       ///< elements from one rhs block to the next
  C **resolvent_ptrs = nullptr;     ///< device array of the Ne resolvent addresses
  C **rhs_ptrs = nullptr;           ///< device array of the Ne rhs addresses
  int *ipiv = nullptr;              ///< Ne x n LU pivots, contiguous as getrfBatched wants
  int *lu_info = nullptr;           ///< Ne getrfBatched infos, one per node
  T *colsum = nullptr;              ///< n: per-column 1-norms of A
  int nodes = 0;                    ///< Ne the layout was sized for
  int k_max = 0;                    ///< widest block filter accepts

  /// @brief Lay the slices out from @p layout; every region is FIXED.
  void carve(WorkspaceLayout &layout, const int n, const int k_cols, const int ne) {
    constexpr std::size_t kAlign = 256;

    const std::size_t nz = static_cast<std::size_t>(n);
    const std::size_t kz = static_cast<std::size_t>(k_cols);
    const std::size_t nez = static_cast<std::size_t>(ne);
    nodes = ne;
    k_max = k_cols;

    // Every block starts on the alignment, which sizeof(C) divides, so the
    // strides are whole elements.
    resolvent_stride = align_up(nz * nz * sizeof(C), kAlign) / sizeof(C);
    rhs_stride = align_up(nz * kz * sizeof(C), kAlign) / sizeof(C);
    resolvents = layout.fixed<C>(resolvent_stride * nez);
    rhs = layout.fixed<C>(rhs_stride * nez);
    resolvent_ptrs = layout.fixed<C *>(nez);
    rhs_ptrs = layout.fixed<C *>(nez);
    ipiv = layout.fixed<int>(nez * nz);
    lu_info = layout.fixed<int>(nez);
    colsum = layout.fixed<T>(nz);
  }
};

/**
 * @brief The feast_resolvent of a dense real symmetric A, of which only the
 *        @p uplo triangle is read. Holds A and its carved workspace, not owning either.
 */
template<calaman::real_fp T>
class DenseResolvent {
public:
  using C = calaman::RealToComplexType<T>;

  /// @param s Carved for this n, a k_max of at least every filter's k, and Ne nodes.
  DenseResolvent(wwr::wwrblasHandle_t cublas_handle, const wwr::wwrblasFillMode_t uplo, const int n,
                 const T *d_A, const int lda, const DenseResolventSlices<T> &s)
      : handle_{cublas_handle}, uplo_{uplo}, n_{n}, d_A_{d_A}, lda_{lda}, s_{s} {}

  /// @brief Y = A X, by symm.
  Status apply(wwr::wwrStream_t /*stream*/, const int k, const T *X, T *Y) {
    wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
    const wwr::extension::ScopedPointerMode mode{handle_, wwr::WWRBLAS_POINTER_MODE_HOST,
                                                 PointerModeStatus{&pm_status}};
    CLM_TRY(pm_status);
    CLM_TRY(wwr::symm<T, int>(handle_, wwr::WWRBLAS_SIDE_LEFT, uplo_, n_, k, &kOne<T>, d_A_, lda_,
                              X, n_, &kZero<T>, Y, n_));
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  /// @brief Build every Z_e I - A and LU-factor them in one batched call.
  ///
  /// Synchronizes @p stream once, to read the per-node LU info: a breakdown
  /// comes back as devinfo_verdict's failure.
  Status prepare(wwr::wwrStream_t stream, const device::FeastContour<T> &contour) {
    if (contour.count < 1 || contour.count > s_.nodes) {
      return wwr::WWRBLAS_STATUS_INVALID_VALUE;
    }
    const bool lower = (uplo_ == wwr::WWRBLAS_FILL_MODE_LOWER);

    device::feast_resolvents(stream, lower, n_, d_A_, lda_, contour, s_.resolvents,
                             s_.resolvent_stride);
    device::feast_pointer_array(stream, s_.resolvents, s_.resolvent_stride, contour.count,
                                s_.resolvent_ptrs);
    device::feast_pointer_array(stream, s_.rhs, s_.rhs_stride, contour.count, s_.rhs_ptrs);
    // A kernel-launch failure is a runtime-domain error, carried as such.
    CLM_TRY(wwr::wwrGetLastError());

    CLM_TRY(wwr::getrfBatched<C>(handle_, n_, s_.resolvent_ptrs, n_, s_.ipiv, s_.lu_info,
                                 contour.count));

    std::array<int, device::kFeastMaxNodes> lu_info{};
    CLM_TRY(wwr::wwrMemcpyAsync(lu_info.data(), s_.lu_info,
                                sizeof(int) * static_cast<std::size_t>(contour.count),
                                wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    for (int e = 0; e < contour.count; ++e) {
      CLM_TRY(devinfo_verdict(lu_info[static_cast<std::size_t>(e)]));
    }
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  /// @brief out = rho(A) Y, from the factors prepare left; @p k <= the slices' k_max.
  Status filter(wwr::wwrStream_t stream, const device::FeastContour<T> &contour, const int k,
                const T *Y, T *out) {
    if (k < 1 || k > s_.k_max || contour.count < 1 || contour.count > s_.nodes) {
      return wwr::WWRBLAS_STATUS_INVALID_VALUE;
    }
    const std::size_t nk = static_cast<std::size_t>(n_) * static_cast<std::size_t>(k);

    device::feast_broadcast(stream, nk, Y, contour.count, s_.rhs, s_.rhs_stride);
    CLM_TRY(wwr::wwrGetLastError());

    // getrsBatched's info is a host int and reports only invalid arguments; the
    // factorization itself was checked by prepare.
    int info = 0;
    CLM_TRY(wwr::getrsBatched<C>(handle_, wwr::WWRBLAS_OP_N, n_, k, s_.resolvent_ptrs, n_, s_.ipiv,
                                 s_.rhs_ptrs, n_, &info, contour.count));
    CLM_TRY(devinfo_verdict(info));

    device::feast_accumulate(stream, nk, contour, s_.rhs, s_.rhs_stride, out);
    CLM_TRY(wwr::wwrGetLastError());
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  /// @brief ||A||_1, exactly, into the device scalar @p d_out: the feast_norm1_hook.
  Status norm1_estimate(wwr::wwrStream_t stream, T *d_out) {
    device::feast_sym_norm1(stream, uplo_ == wwr::WWRBLAS_FILL_MODE_LOWER, n_, d_A_, lda_,
                            s_.colsum, d_out);
    CLM_TRY(wwr::wwrGetLastError());
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  /// @brief The dimension n, and the widest block and most nodes the slices fit.
  int dim() const noexcept { return n_; }
  int k_max() const noexcept { return s_.k_max; }
  int shifts_max() const noexcept { return s_.nodes; }

private:
  wwr::wwrblasHandle_t handle_;
  wwr::wwrblasFillMode_t uplo_;
  int n_;
  const T *d_A_;
  int lda_;
  DenseResolventSlices<T> s_;
};

static_assert(slices_for<DenseResolventSlices<float>, int, int, int>);
static_assert(feast_resolvent<DenseResolvent<float>, float>);
static_assert(feast_resolvent<DenseResolvent<double>, double>);
static_assert(!feast_resolvent<DenseResolvent<double>, float>);
static_assert(feast_norm1_hook<DenseResolvent<double>, double>);
static_assert(!feast_residual_hook<DenseResolvent<double>, double>, "exact solves: plain form");

} // namespace calaman
