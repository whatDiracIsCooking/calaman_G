/**
 * @file ritz.cppm
 * @brief The calaman.ritz module -- the Ritz-pair bookkeeping every projection
 *        eigensolver shares: select, classify, rotate
 *
 * ritz_select picks the wanted positions of an ascending spectrum (by count
 * alone, or by its values for largest_magnitude);
 * classify_ritz turns values plus residual estimates into a RitzSelection under
 * a convergence predicate; ritz_rotate forms the Ritz vectors C = B S with one
 * gemm. calaman.lanczos, calaman.davidson and calaman.feast import it.
 *
 * Signatures are raw pointers plus leading dimensions, never a solver's slices
 * struct: each solver owns its own workspace carve (the `workspace` skill).
 * One convergence predicate, shared by lanczos and davidson: residual <=
 * tolerance * max(|value|, scale), scale the projected matrix's 2-norm
 * (docs/architecture.md §8; feast's backward error is the other meaning).
 *
 * Usage:
 *   import calaman.ritz;
 *   const std::vector<int> index = ritz_select(RitzWhich::smallest, ncv, nev);
 *   auto sel = classify_ritz<double>(values, residuals, tolerance, t_norm);
 *   CLM_TRY(ritz_rotate<double>(blas, n, ncv, nev, d_V, n, d_S, ncv, d_X, n));
 */

module;

// CLM_TRY / CLM_REQUIRE -- macros, so they arrive by #include in the GMF; they
// need calaman::Status visible at expansion, which the export import supplies.
#include "error_handling/error_macros.h"

export module calaman.ritz;

import std;
import wwr.blas;                      // wwrblasHandle_t, WWRBLAS_*
import wwr.wrappers.blas;             // gemm
import wwr.extension.blas;            // ScopedPointerMode
import calaman.common;                // kOne, kZero, real_fp
export import calaman.error_handling; // Status, PointerModeStatus

namespace calaman::detail {

/// @brief Positions [0, low) and [available - (count - low), available).
inline std::vector<int> ritz_ends(const int available, const int count, const int low) {
  std::vector<int> index;
  index.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < low; ++i) {
    index.push_back(i);
  }
  for (int i = available - (count - low); i < available; ++i) {
    index.push_back(i);
  }
  return index;
}

} // namespace calaman::detail

export namespace calaman {

/// @brief Which end of an ascending spectrum a solver converges.
enum class RitzWhich {
  smallest,          ///< the count algebraically smallest values
  largest,           ///< the count algebraically largest values
  both_ends,         ///< ceil(count / 2) from the top, floor(count / 2) from the bottom
  largest_magnitude, ///< the count largest |value|; needs the values (ritz_select's span form)
};

/// @brief Classified Ritz pairs: values, residual estimates and convergence flags.
template<calaman::real_fp T>
struct RitzSelection {
  std::vector<int> index;      ///< source positions, if recorded (classify_ritz leaves it empty)
  std::vector<T> values;       ///< Ritz values
  std::vector<T> residuals;    ///< their residual estimates
  std::vector<bool> converged; ///< residuals[i] within classify_ritz's threshold
  int converged_count = 0;

  bool all_converged() const { return converged_count == static_cast<int>(values.size()); }
};

/// @brief The @p count positions, ascending, of an ascending @p available-long
///        spectrum that @p which wants. For count2 >= count1 at one @p which,
///        the count2 selection contains the count1 one.
/// @return Empty unless 1 <= count <= available; always empty for
///         largest_magnitude, which needs the values (the span overload).
inline std::vector<int> ritz_select(const RitzWhich which, const int available, const int count) {
  if (count < 1 || count > available) {
    return {};
  }
  switch (which) {
  case RitzWhich::smallest:
    return detail::ritz_ends(available, count, count);
  case RitzWhich::largest:
    return detail::ritz_ends(available, count, 0);
  case RitzWhich::both_ends:
    return detail::ritz_ends(available, count, count / 2);
  case RitzWhich::largest_magnitude:
    break;
  }
  return {};
}

/// @brief ritz_select over the ascending @p values themselves, the only form
///        that serves largest_magnitude: |value| ties go to the top end, so
///        selections still nest. Every other @p which ignores the values.
/// @return Empty unless 1 <= count <= values.size().
template<calaman::real_fp T>
std::vector<int> ritz_select(const RitzWhich which, const std::span<const T> values,
                             const int count) {
  const int available = static_cast<int>(values.size());
  if (which != RitzWhich::largest_magnitude) {
    return ritz_select(which, available, count);
  }
  if (count < 1 || count > available) {
    return {};
  }
  // Ascending, so |value| is largest at the ends: merge inward, one per step.
  int low = 0;
  for (int high = 0; low + high < count;) {
    const auto b = static_cast<std::size_t>(low);
    const auto t = static_cast<std::size_t>(available - 1 - high);
    if (std::abs(values[b]) > std::abs(values[t])) {
      ++low;
    } else {
      ++high;
    }
  }
  return detail::ritz_ends(available, count, low);
}

/// @brief Classify each pair: converged iff residuals[i] <= tolerance *
///        max(|values[i]|, @p scale), @p scale the projected matrix's 2-norm.
///        Host only; index is left empty.
/// @return Empty when @p values and @p residuals differ in length.
template<calaman::real_fp T>
RitzSelection<T> classify_ritz(const std::span<const T> values, const std::span<const T> residuals,
                               const T tolerance, const T scale) {
  RitzSelection<T> sel;
  if (values.size() != residuals.size()) {
    return sel;
  }
  sel.values.assign(values.begin(), values.end());
  sel.residuals.assign(residuals.begin(), residuals.end());
  sel.converged.reserve(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    const bool ok = residuals[i] <= tolerance * std::max(std::abs(values[i]), scale);
    sel.converged.push_back(ok);
    sel.converged_count += ok ? 1 : 0;
  }
  return sel;
}

/**
 * @brief Ritz vectors C = B(:, 0:k) * S(0:k, 0:count), one gemm in HOST pointer
 *        mode. The handle's mode is scoped and restored, so a caller already in
 *        host mode pays only a get/set pair, no sync.
 *
 * @param b Device, n x k, ld @p ldb >= n: the basis.
 * @param s Device, k x count, ld @p lds >= k: the projected eigenvectors.
 * @param c Out: device, n x count, ld @p ldc >= n; must not overlap @p b or @p s.
 * @return INVALID_VALUE for n < 1, count outside [1, k], a bad leading
 *         dimension or a null pointer; otherwise the first failing call's status.
 */
template<calaman::real_fp T>
Status ritz_rotate(wwr::wwrblasHandle_t blas_handle, const int n, const int k, const int count,
                   const T *b, const int ldb, const T *s, const int lds, T *c, const int ldc) {
  CLM_REQUIRE(b != nullptr && s != nullptr && c != nullptr && n >= 1 && count >= 1 && count <= k &&
                  ldb >= n && lds >= k && ldc >= n,
              wwr::WWRBLAS_STATUS_INVALID_VALUE);
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode mode{blas_handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                               PointerModeStatus{&pm_status}};
  CLM_TRY(pm_status);
  CLM_TRY((wwr::gemm<T, int>(blas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, count, k,
                             &kOne<T>, b, ldb, s, lds, &kZero<T>, c, ldc)));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
