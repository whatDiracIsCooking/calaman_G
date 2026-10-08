/**
 * @file krylov_resolvent.cppm
 * @brief KrylovResolvent, the matrix-free feast_resolvent model: the Ne shifted
 *        solves of the filter by calaman.shifted_cocg over any linear_operator
 *
 * The :krylov_resolvent partition of calaman.feast.
 *
 *   filter(Y) = sum_e Re[ w_e X_e ],   (Z_e I - A) X_e = Y by shifted_cocg
 *
 * All Ne shifts share one Krylov space, so a filter costs one k-column apply
 * per inner step for every node together. prepare has nothing to factor.
 *
 * The inner solves are inexact: each column meets the options' relative
 * residual tolerance, so filter agrees with DenseResolvent's to that tolerance
 * scaled by sum_e |w_e| / Im Z_e. An inner solve that stops short of it
 * (MaxIterations or NumericalFailure) fails filter with EXECUTION_FAILED --
 * feast then stops with NumericalFailure -- and last_solve() says why.
 *
 * No norm1_estimate hook: the driver estimates ||A||_1 by lacn2 over apply.
 */

module;

#include "feast_bridge.h"

// CLM_TRY / CLM_REQUIRE -- macros, so they arrive by #include in the global
// module fragment; they need calaman::Status visible at expansion, which the
// export import below supplies.
#include "error_handling/error_macros.h"

export module calaman.feast:krylov_resolvent;

import std;
import wwr.runtime_api;               // wwrStream_t, wwrGetLastError
import wwr.blas;                      // WWRBLAS_STATUS_*
import calaman.common;                // real_fp, WorkspaceLayout, carve_workspace, slices_for
import :resolvent;                    // feast_resolvent, feast_norm1_hook
export import calaman.shifted_cocg;   // shifted_cocg, its options, info and slices
export import calaman.error_handling; // Status

export namespace calaman {

/// @brief KrylovResolvent's workspace: shifted_cocg's, then the split X_e it solves into.
template<calaman::real_fp T>
struct KrylovResolventSlices {
  ShiftedCocgSlices<T> cocg;
  T *xr = nullptr; ///< Ne blocks of n x k: Re X_e
  T *xi = nullptr; ///< Ne blocks of n x k: Im X_e

  /// @brief Lay the slices out from @p layout; every region is FIXED.
  void carve(WorkspaceLayout &layout, const int n, const int k_max, const int ne) {
    cocg.carve(layout, n, k_max, ne);
    const std::size_t len = static_cast<std::size_t>(n) * static_cast<std::size_t>(k_max) *
                            static_cast<std::size_t>(ne);
    xr = layout.fixed<T>(len);
    xi = layout.fixed<T>(len);
  }
};

/**
 * @brief Carve @p d_work into KrylovResolventSlices for an n-dimensional operator,
 *        filter blocks up to @p k_max columns, and @p ne quadrature nodes.
 *
 * @param d_work Workspace, or null to size it only. @p slices, @p lwork_bytes: out, may be null.
 */
template<calaman::real_fp T>
Status make_krylov_resolvent_slices(const int n, const int k_max, const int ne, void *d_work,
                                    KrylovResolventSlices<T> *slices, std::size_t *lwork_bytes) {
  CLM_REQUIRE(shifted_cocg_shape_ok(n, k_max, ne) && ne <= device::kFeastMaxNodes,
              wwr::WWRBLAS_STATUS_INVALID_VALUE);
  const std::size_t bytes = carve_workspace(d_work, slices, n, k_max, ne);
  if (lwork_bytes != nullptr) {
    *lwork_bytes = bytes;
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief Device workspace, in bytes, that KrylovResolvent needs at (n, k_max, ne).
/// @return INVALID_VALUE for a null @p lwork_bytes, or a shape outside 1 <= ne <= 8.
template<calaman::real_fp T>
Status krylov_resolvent_bufferSize(const int n, const int k_max, const int ne,
                                   std::size_t *lwork_bytes) {
  CLM_REQUIRE(lwork_bytes != nullptr, wwr::WWRBLAS_STATUS_INVALID_VALUE);
  return make_krylov_resolvent_slices<T>(n, k_max, ne, nullptr, nullptr, lwork_bytes);
}

/**
 * @brief The feast_resolvent of the real symmetric A that @p Op applies, by
 *        shifted_cocg. Holds @p op and the carved workspace, owning neither.
 */
template<class Op, calaman::real_fp T>
  requires linear_operator<Op, T>
class KrylovResolvent {
public:
  /// @param s Carved for this n, a k_max of at least every filter's k, and Ne nodes.
  /// @param options The inner solves' relative tolerance and step cap.
  KrylovResolvent(Op &op, const int n, const KrylovResolventSlices<T> &s,
                  const ShiftedCocgOptions<T> &options = {})
      : op_{&op}, n_{n}, s_{s}, options_{options} {}

  /// @brief Y = A X, by the operator.
  Status apply(wwr::wwrStream_t stream, const int k, const T *X, T *Y) {
    return op_->apply(stream, k, X, Y);
  }

  /// @brief Checks the contour fits the workspace; there is nothing to factor.
  Status prepare(wwr::wwrStream_t /*stream*/, const device::FeastContour<T> &contour) {
    CLM_REQUIRE(contour.count >= 1 && contour.count <= s_.cocg.shifts_max,
                wwr::WWRBLAS_STATUS_INVALID_VALUE);
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  /// @brief out = sum_e Re[ w_e (Z_e I - A)^{-1} Y ], each solve to the options'
  ///        tolerance; EXECUTION_FAILED if one stops short (see last_solve).
  Status filter(wwr::wwrStream_t stream, const device::FeastContour<T> &contour, const int k,
                const T *Y, T *out) {
    CLM_REQUIRE(k >= 1 && k <= s_.cocg.k_max && contour.count >= 1 &&
                    contour.count <= s_.cocg.shifts_max,
                wwr::WWRBLAS_STATUS_INVALID_VALUE);
    const auto count = static_cast<std::size_t>(contour.count);
    CLM_TRY(shifted_cocg<T>(stream, *op_, n_, k, std::span<const T>{contour.zr, count},
                            std::span<const T>{contour.zi, count}, Y, s_.xr, s_.xi, s_.cocg,
                            &last_, options_));
    if (!converged(last_)) {
      return wwr::WWRBLAS_STATUS_EXECUTION_FAILED;
    }
    const std::size_t nk = static_cast<std::size_t>(n_) * static_cast<std::size_t>(k);
    device::feast_accumulate_split(stream, nk, contour, s_.xr, s_.xi, nk, out);
    CLM_TRY(wwr::wwrGetLastError());
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  /// @brief What the last filter's shifted_cocg did: iterations, reason, per-shift residuals.
  const ShiftedCocgInfo<T> &last_solve() const noexcept { return last_; }

private:
  Op *op_;
  int n_;
  KrylovResolventSlices<T> s_;
  ShiftedCocgOptions<T> options_;
  ShiftedCocgInfo<T> last_{};
};

} // namespace calaman

namespace calaman::detail {

/// The minimal operator, to check the model against the concepts. Declared only.
template<class T>
struct KrylovCheckOperator {
  Status apply(wwr::wwrStream_t stream, int k, const T *X, T *Y);
};
static_assert(slices_for<KrylovResolventSlices<double>, int, int, int>);
static_assert(feast_resolvent<KrylovResolvent<KrylovCheckOperator<float>, float>, float>);
static_assert(feast_resolvent<KrylovResolvent<KrylovCheckOperator<double>, double>, double>);
static_assert(!feast_norm1_hook<KrylovResolvent<KrylovCheckOperator<double>, double>, double>);

} // namespace calaman::detail
