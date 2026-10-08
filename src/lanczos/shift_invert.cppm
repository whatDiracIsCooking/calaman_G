/**
 * @file shift_invert.cppm
 * @brief Shift-invert Lanczos: the eigenpairs of a symmetric A nearest a shift
 *
 * The :shift_invert partition of calaman.lanczos. lanczos_shift_invert_solve
 * runs lanczos_solve unchanged on a shifted_operator -- (A - sigma I)^{-1}, e.g.
 * calaman.shift_invert's DenseShiftInvert -- for the largest_magnitude Ritz
 * values theta, then back-transforms each to lambda = sigma + 1/theta, ascending.
 * Exact solves only (docs/architecture.md §9). No workspace of its own: the
 * operator owns its factor, the driver its LanczosSlices.
 *
 * Usage:
 *   DenseShiftInvert<double> op{blas, solver, Uplo::L, n, d_A, n, sigma, op_slices};
 *   CLM_TRY(op.prepare(stream));
 *   CLM_TRY(lanczos_shift_invert_solve<double>(blas, solver, stream, n, nev, ncv, s, op,
 *                                              d_lambda, d_X, &info));
 */

module;

// CLM_TRY / CLM_REQUIRE -- macros, so they arrive by #include in the GMF; they
// need calaman::Status visible at expansion, which the export import supplies.
#include "error_handling/error_macros.h"

#include <cstddef>

export module calaman.lanczos:shift_invert;

import std;
import wwr.blas;          // wwrblasHandle_t, WWRBLAS_STATUS_*
import wwr.solver;        // wwrsolverDnHandle_t
import wwr.runtime_api;   // wwrStream_t, wwrMemcpyAsync, wwrStreamSynchronize
import wwr.wrappers.blas; // swap
import :buffer_size;      // LanczosSlices
import :solve;            // lanczos_solve
import :types;            // LanczosOptions, LanczosInfo, LanczosWhich, linear_operator
import calaman.common;    // real_fp
export import calaman.error_handling; // Status

export namespace calaman {

/// @brief A linear_operator for (A - sigma I)^{-1} that reports its shift:
///        what lanczos_shift_invert_solve back-transforms with.
template<typename Op, typename T>
concept shifted_operator = linear_operator<Op, T> && requires(const Op &op) {
  { op.sigma() } -> std::convertible_to<T>;
};

/// @brief Overwrite the Ritz values @p values (theta) with lambda = sigma + 1/theta,
///        sorted ascending. Host only.
/// @return The source of each output: out[j] came from position order[j].
template<calaman::real_fp T>
std::vector<int> shift_invert_back_transform(const T sigma, const std::span<T> values) {
  std::vector<T> lambda(values.size());
  for (std::size_t j = 0; j < values.size(); ++j) {
    lambda[j] = sigma + T{1} / values[j];
  }
  std::vector<int> order(values.size());
  std::iota(order.begin(), order.end(), 0);
  std::ranges::stable_sort(order, [&lambda](const int a, const int b) {
    return std::strong_order(lambda[static_cast<std::size_t>(a)],
                             lambda[static_cast<std::size_t>(b)]) < 0;
  });
  for (std::size_t j = 0; j < values.size(); ++j) {
    values[j] = lambda[static_cast<std::size_t>(order[j])];
  }
  return order;
}

/**
 * @brief The @p nev eigenpairs of A nearest op.sigma(): lanczos_solve on @p op
 *        (ready to apply) for largest_magnitude, then shift_invert_back_transform.
 *        options.tolerance, and verify_residuals, judge theta against @p op, not
 *        lambda against A. Otherwise as lanczos_solve, with lambda for theta.
 * @param eigenvalues_out  Out: the nev lambdas, ascending, device.
 * @param eigenvectors_out Out: n x nev (ld n), in the same order; null to skip.
 */
template<calaman::real_fp T, shifted_operator<T> Op>
Status lanczos_shift_invert_solve(wwr::wwrblasHandle_t blas_handle,
                                  wwr::wwrsolverDnHandle_t solver_handle, wwr::wwrStream_t stream,
                                  const int n, const int nev, const int ncv,
                                  const LanczosSlices<T> &s, Op &op, T *eigenvalues_out,
                                  T *eigenvectors_out, LanczosInfo *info,
                                  const LanczosOptions<T> &options = {}) {
  CLM_TRY(lanczos_solve<T>(blas_handle, solver_handle, stream, n, nev, ncv,
                           LanczosWhich::largest_magnitude, s, op, eigenvalues_out,
                           eigenvectors_out, info, options));
  const LanczosStopReason reason = info->reason;
  info->reason = LanczosStopReason::NumericalFailure;

  const auto nevz = static_cast<std::size_t>(nev);
  std::vector<T> values(nevz);
  CLM_TRY(wwr::wwrMemcpyAsync(values.data(), eigenvalues_out, sizeof(T) * nevz,
                              wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  const std::vector<int> order = shift_invert_back_transform<T>(static_cast<T>(op.sigma()), values);

  if (eigenvectors_out != nullptr) {
    // Apply the permutation in place, one column swap per misplaced column:
    // at[c] is the source pair now in column c, where[p] the column holding p.
    std::vector<int> at(nevz);
    std::iota(at.begin(), at.end(), 0);
    std::vector<int> where = at;
    const auto nz = static_cast<std::size_t>(n);
    for (std::size_t j = 0; j < nevz; ++j) {
      const int want = order[j];
      const int from = where[static_cast<std::size_t>(want)];
      if (from == static_cast<int>(j)) {
        continue;
      }
      CLM_TRY((wwr::swap<T, int>(blas_handle, n, eigenvectors_out + j * nz, 1,
                                 eigenvectors_out + static_cast<std::size_t>(from) * nz, 1)));
      const int displaced = at[j];
      at[j] = want;
      at[static_cast<std::size_t>(from)] = displaced;
      where[static_cast<std::size_t>(want)] = static_cast<int>(j);
      where[static_cast<std::size_t>(displaced)] = from;
    }
  }
  // values is host-local: the sync keeps it alive until the copy lands.
  CLM_TRY(wwr::wwrMemcpyAsync(eigenvalues_out, values.data(), sizeof(T) * nevz,
                              wwr::wwrMemcpyHostToDevice, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  info->reason = reason;
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
