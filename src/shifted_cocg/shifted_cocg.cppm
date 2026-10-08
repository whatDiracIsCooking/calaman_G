/**
 * @file shifted_cocg.cppm
 * @brief The calaman.shifted_cocg module -- multi-shift block COCG for the
 *        complex-symmetric shifted systems (z_e I - A) X_e = B
 *
 * A is real symmetric, known only as a linear_operator; B is real n x k. Every
 * shift solves over the one Krylov space K(A, b_j) (shift invariance), so each
 * step costs one k-column operator apply for all ne shifts. COCG uses the
 * bilinear form x^T y, right for z I - A (complex symmetric, not Hermitian).
 * It runs here in its Lanczos form: a real three-term Lanczos per column, with
 * the per-shift COCG coefficients as complex scalar recurrences (D-Lanczos),
 * which never break down for Im z_e > 0; columns batch into one block apply.
 * X_e is stored split, Re and Im; shifted_cocg_accumulate stores instead only
 * a weighted sum of the X_e, one n x k block in place of ne.
 *
 * A pair (shift, column) stops updating once its relative residual estimate
 * beta_{m+1} |zeta_m / eta_m| / ||b_j|| meets the tolerance. No
 * reorthogonalization: in floating point the estimate stays a residual bound
 * of the computed iterate, but convergence may take more than n steps.
 * Non-convergence is an outcome, not an error (src/iterative/README.md).
 *
 * Convergence is read back (one host sync) at the start, every check_interval
 * steps and at the last step; the steps between enqueue without a sync. A pair
 * freezes at the step it converged, so results do not depend on the interval;
 * only iterations rounds up to it. Owns no handle: @p op enqueues on @p stream.
 *
 * Usage:
 *   import calaman.shifted_cocg;
 *
 *   std::size_t bytes = 0;
 *   calaman::shifted_cocg_bufferSize<double>(n, k, ne, &bytes);
 *   calaman::ShiftedCocgSlices<double> s;
 *   calaman::make_shifted_cocg_slices<double>(n, k, ne, d_work, &s, nullptr);
 *   calaman::ShiftedCocgInfo<double> info;
 *   calaman::shifted_cocg<double>(stream, op, n, k, zr, zi, d_B, d_Xr, d_Xi, s, &info);
 */

module;

// CLM_TRY / CLM_REQUIRE -- macros, so they arrive by #include in the GMF; they
// need calaman::Status visible at expansion, which the export import supplies.
#include "error_handling/error_macros.h"
#include "shifted_cocg_bridge.h"

#include <cstddef>

export module calaman.shifted_cocg;

import std;
import wwr.blas;        // WWRBLAS_STATUS_*
import wwr.runtime_api; // wwrStream_t, wwrMemcpyAsync, wwrMemsetAsync, wwrStreamSynchronize
import calaman.common;  // WorkspaceLayout, carve_workspace, slices_for, real_fp
export import calaman.error_handling;  // Status
export import calaman.iterative;       // IterationInfo, stop_reason, converged
export import calaman.linear_operator; // linear_operator

export namespace calaman {

/// @brief Stopping knobs for one shifted_cocg call.
template<calaman::real_fp T>
struct ShiftedCocgOptions {
  /// @brief Relative: pair (e, j) converges once its residual estimate
  ///        ||b_j - (z_e I - A) x_ej||_2 <= tolerance * ||b_j||_2.
  T tolerance = T{1e-8};
  /// @brief Lanczos steps (block operator applies) before giving up.
  int max_iterations = 1000;
  /// @brief Steps between convergence read-backs (host syncs); at least 1.
  int check_interval = 8;
};

/// @brief Why shifted_cocg stopped.
enum class ShiftedCocgStopReason {
  Converged,        ///< every pair met the tolerance
  MaxIterations,    ///< max_iterations steps ran out
  NumericalFailure, ///< a residual estimate turned NaN (a non-finite A or B)
};
static_assert(stop_reason<ShiftedCocgStopReason>);

/// @brief What shifted_cocg did: iterations are Lanczos steps (block applies),
///        past convergence rounded up to a check (see check_interval).
template<calaman::real_fp T>
struct ShiftedCocgInfo : IterationInfo<ShiftedCocgStopReason> {
  /// @brief Per shift: steps until all k columns converged, else iterations.
  std::vector<int> shift_iterations;
  /// @brief Per shift: largest relative residual estimate over the columns.
  std::vector<T> shift_residual;
  /// @brief Convergence read-backs, each one host sync; the start's included.
  int convergence_checks = 0;
};

/// @brief Pointers into the solver's single device workspace buffer.
template<calaman::real_fp T>
struct ShiftedCocgSlices {
  T *v = nullptr;           ///< 3 blocks of n x k: the Lanczos v_{m-1}, v_m, v_{m+1}
  T *p_re = nullptr;        ///< ne blocks of n x k: Re of the search directions
  T *p_im = nullptr;        ///< ne blocks of n x k: Im of the search directions
  T *shift_re = nullptr;    ///< ne: Re z_e
  T *shift_im = nullptr;    ///< ne: Im z_e
  T *columns = nullptr;     ///< 4 x k: alpha, two beta slots, ||b_j||
  T *pair_state = nullptr;  ///< 7 x ne x k: eta, zeta, 1/eta (re, im), residual
  int *done_step = nullptr; ///< ne x k: -1 iterating, else the converged step
  int n = 0;                ///< operator dimension the layout was sized for
  int k_max = 0;            ///< widest block of right-hand sides
  int shifts_max = 0;       ///< most shifts

  /// @brief Lay the slices out from @p layout; every region is FIXED.
  void carve(WorkspaceLayout &layout, const int dim, const int k_cols, const int ne) {
    const std::size_t block = static_cast<std::size_t>(dim) * static_cast<std::size_t>(k_cols);
    const std::size_t nez = static_cast<std::size_t>(ne);
    const std::size_t kz = static_cast<std::size_t>(k_cols);
    n = dim;
    k_max = k_cols;
    shifts_max = ne;
    v = layout.fixed<T>(3 * block);
    p_re = layout.fixed<T>(nez * block);
    p_im = layout.fixed<T>(nez * block);
    shift_re = layout.fixed<T>(nez);
    shift_im = layout.fixed<T>(nez);
    columns = layout.fixed<T>(4 * kz);
    pair_state = layout.fixed<T>(7 * nez * kz);
    done_step = layout.fixed<int>(nez * kz);
  }
};

/// @brief Whether (n, k, ne) is a shape the solver accepts: all at least 1.
constexpr bool shifted_cocg_shape_ok(const int n, const int k, const int ne) noexcept {
  return n >= 1 && k >= 1 && ne >= 1;
}

/**
 * @brief Carve @p d_work into ShiftedCocgSlices for an n-dimensional operator,
 *        up to @p k right-hand sides and @p ne shifts.
 *
 * @param d_work      Workspace, or null to size it only.
 * @param slices      Out, may be null.
 * @param lwork_bytes Out, may be null: the bytes the layout needs.
 */
template<calaman::real_fp T>
Status make_shifted_cocg_slices(const int n, const int k, const int ne, void *d_work,
                                ShiftedCocgSlices<T> *slices, std::size_t *lwork_bytes) {
  CLM_REQUIRE(shifted_cocg_shape_ok(n, k, ne), wwr::WWRBLAS_STATUS_INVALID_VALUE);
  const std::size_t bytes = carve_workspace(d_work, slices, n, k, ne);
  if (lwork_bytes != nullptr) {
    *lwork_bytes = bytes;
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief Device workspace, in bytes, that shifted_cocg needs at (n, k, ne).
/// @return INVALID_VALUE for a null @p lwork_bytes or a rejected shape.
template<calaman::real_fp T>
Status shifted_cocg_bufferSize(const int n, const int k, const int ne, std::size_t *lwork_bytes) {
  CLM_REQUIRE(lwork_bytes != nullptr, wwr::WWRBLAS_STATUS_INVALID_VALUE);
  return make_shifted_cocg_slices<T>(n, k, ne, nullptr, nullptr, lwork_bytes);
}

} // namespace calaman

namespace calaman::detail {

/// The shared solve: validates, runs @p clear once the arguments pass, then per
/// step @p update(step, v_cur, beta_cur, pairs) enqueues the P_e / output update.
template<calaman::real_fp T, linear_operator<T> Op, class Clear, class Update>
Status shifted_cocg_run(wwr::wwrStream_t stream, Op &op, const int n, const int k,
                        std::span<const T> zr, std::span<const T> zi, const T *d_B,
                        const ShiftedCocgSlices<T> &s, ShiftedCocgInfo<T> *info,
                        const ShiftedCocgOptions<T> &options, Clear clear, Update update) {
  const int ne = static_cast<int>(zr.size());
  CLM_REQUIRE(info != nullptr && d_B != nullptr, wwr::WWRBLAS_STATUS_INVALID_VALUE);
  CLM_REQUIRE(shifted_cocg_shape_ok(n, k, ne) && zi.size() == zr.size(),
              wwr::WWRBLAS_STATUS_INVALID_VALUE);
  CLM_REQUIRE(s.v != nullptr && s.n == n && k <= s.k_max && ne <= s.shifts_max,
              wwr::WWRBLAS_STATUS_INVALID_VALUE);
  CLM_REQUIRE(options.tolerance >= T{0} && options.max_iterations >= 0 &&
                  options.check_interval >= 1,
              wwr::WWRBLAS_STATUS_INVALID_VALUE);
  for (int e = 0; e < ne; ++e) {
    const auto ez = static_cast<std::size_t>(e);
    CLM_REQUIRE(std::isfinite(zr[ez]) && std::isfinite(zi[ez]) && zi[ez] > T{0},
                wwr::WWRBLAS_STATUS_INVALID_VALUE);
  }

  const std::size_t block = static_cast<std::size_t>(n) * static_cast<std::size_t>(k);
  const std::size_t pairs_count = static_cast<std::size_t>(ne) * static_cast<std::size_t>(k);
  const std::size_t kz = static_cast<std::size_t>(k);
  const std::size_t pz = pairs_count;
  const device::CocgPairs<T> pairs{
      s.pair_state,          s.pair_state + pz,     s.pair_state + 2 * pz, s.pair_state + 3 * pz,
      s.pair_state + 4 * pz, s.pair_state + 5 * pz, s.pair_state + 6 * pz, s.done_step};
  T *alpha = s.columns;
  T *beta_cur = s.columns + kz;
  T *beta_next = s.columns + 2 * kz;
  T *bnorm = s.columns + 3 * kz;
  T *v_prev = s.v;
  T *v_cur = s.v + block;
  T *v_next = s.v + 2 * block;

  const std::size_t x_bytes = sizeof(T) * block * static_cast<std::size_t>(ne);
  CLM_TRY(clear(x_bytes));
  CLM_TRY(wwr::wwrMemsetAsync(s.p_re, 0, x_bytes, stream));
  CLM_TRY(wwr::wwrMemsetAsync(s.p_im, 0, x_bytes, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(s.shift_re, zr.data(), sizeof(T) * zr.size(),
                              wwr::wwrMemcpyHostToDevice, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(s.shift_im, zi.data(), sizeof(T) * zi.size(),
                              wwr::wwrMemcpyHostToDevice, stream));
  device::cocg_start<T>(stream, n, k, ne, d_B, v_prev, v_cur, beta_cur, bnorm, pairs);
  CLM_TRY(wwr::wwrGetLastError());

  std::vector<int> done(pairs_count);
  std::vector<T> res(pairs_count);
  // Reads back the pair state; true when every pair is done. Syncs the stream.
  const auto read_back = [&]() -> Status {
    ++info->convergence_checks;
    CLM_TRY(wwr::wwrMemcpyAsync(done.data(), s.done_step, sizeof(int) * pairs_count,
                                wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(wwr::wwrMemcpyAsync(res.data(), pairs.res, sizeof(T) * pairs_count,
                                wwr::wwrMemcpyDeviceToHost, stream));
    return wwr::wwrStreamSynchronize(stream);
  };
  const auto all_done = [&] {
    return std::ranges::all_of(done, [](const int d) { return d != -1; });
  };

  info->iterations = 0;
  info->reason = ShiftedCocgStopReason::MaxIterations;
  info->convergence_checks = 0;
  CLM_TRY(read_back());
  for (int step = 1; !all_done() && step <= options.max_iterations; ++step) {
    CLM_TRY(op.apply(stream, k, v_cur, v_next));
    device::cocg_lanczos<T>(stream, n, k, v_prev, v_cur, v_next, beta_cur, alpha, beta_next);
    device::cocg_shift<T>(stream, k, ne, step, options.tolerance, s.shift_re, s.shift_im, alpha,
                          beta_cur, beta_next, bnorm, pairs);
    update(step, static_cast<const T *>(v_cur), static_cast<const T *>(beta_cur), pairs);
    CLM_TRY(wwr::wwrGetLastError());
    info->iterations = step;
    // v_{m+1} becomes v_m, and beta_{m+1} beta_m; the old v_{m-1} is reused.
    std::swap(v_prev, v_cur);
    std::swap(v_cur, v_next);
    std::swap(beta_cur, beta_next);
    // Between checks all_done() reads stale flags: the loop runs on to the next.
    if (step % options.check_interval != 0 && step != options.max_iterations) {
      continue;
    }
    CLM_TRY(read_back());
    if (std::ranges::any_of(res, [](const T r) { return std::isnan(r); })) {
      info->reason = ShiftedCocgStopReason::NumericalFailure;
      break;
    }
  }
  if (info->reason != ShiftedCocgStopReason::NumericalFailure && all_done()) {
    info->reason = ShiftedCocgStopReason::Converged;
  }

  info->shift_iterations.assign(static_cast<std::size_t>(ne), 0);
  info->shift_residual.assign(static_cast<std::size_t>(ne), T{0});
  for (std::size_t p = 0; p < pairs_count; ++p) {
    const std::size_t e = p / kz;
    const int steps = done[p] == -1 ? info->iterations : done[p];
    info->shift_iterations[e] = std::max(info->shift_iterations[e], steps);
    info->shift_residual[e] = std::max(info->shift_residual[e], res[p]);
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman::detail

export namespace calaman {

/**
 * @brief Solve (z_e I - A) X_e = B for every shift z_e = zr[e] + i zi[e].
 *
 * Returns success whenever it stops on its own condition (read @p info);
 * INVALID_VALUE for a bad argument, any Im z_e <= 0 included.
 * @param op     Applies the real symmetric A; called with k columns per step.
 * @param d_B    n x k, device, ld n. Never written.
 * @param d_Xr   Out: ne blocks of n x k (ld n), n * k apart: Re X_e. Likewise @p d_Xi.
 * @param s      Carved for this n, a k_max >= k and at least zr.size() shifts.
 */
template<calaman::real_fp T, linear_operator<T> Op>
Status shifted_cocg(wwr::wwrStream_t stream, Op &op, const int n, const int k,
                    std::span<const T> zr, std::span<const T> zi, const T *d_B, T *d_Xr, T *d_Xi,
                    const ShiftedCocgSlices<T> &s, ShiftedCocgInfo<T> *info,
                    const ShiftedCocgOptions<T> &options = {}) {
  CLM_REQUIRE(d_Xr != nullptr && d_Xi != nullptr, wwr::WWRBLAS_STATUS_INVALID_VALUE);
  const int ne = static_cast<int>(zr.size());
  const auto clear = [&](const std::size_t x_bytes) -> Status {
    CLM_TRY(wwr::wwrMemsetAsync(d_Xr, 0, x_bytes, stream));
    return wwr::wwrMemsetAsync(d_Xi, 0, x_bytes, stream);
  };
  const auto update = [&](const int step, const T *v_cur, const T *beta_cur,
                          const device::CocgPairs<T> &pairs) {
    device::cocg_update<T>(stream, n, k, ne, step, v_cur, beta_cur, pairs, s.p_re, s.p_im, d_Xr,
                           d_Xi);
  };
  return detail::shifted_cocg_run<T>(stream, op, n, k, zr, zi, d_B, s, info, options, clear,
                                     update);
}

/**
 * @brief shifted_cocg without X_e: d_out(:, j) += sum_e Re[ c_ej X_e(:, j) ], the
 *        sum built step by step, so no ne blocks of X are stored.
 *
 * Arguments and returns as shifted_cocg. A partial solve accumulates its partial X_e.
 * @param d_cr  ne x k, device: Re c_ej at e * k + j. Likewise @p d_ci, Im c_ej.
 * @param d_out n x k, device, ld n, added to; must not overlap @p d_B.
 */
template<calaman::real_fp T, linear_operator<T> Op>
Status shifted_cocg_accumulate(wwr::wwrStream_t stream, Op &op, const int n, const int k,
                               std::span<const T> zr, std::span<const T> zi, const T *d_B,
                               const T *d_cr, const T *d_ci, T *d_out,
                               const ShiftedCocgSlices<T> &s, ShiftedCocgInfo<T> *info,
                               const ShiftedCocgOptions<T> &options = {}) {
  CLM_REQUIRE(d_cr != nullptr && d_ci != nullptr && d_out != nullptr,
              wwr::WWRBLAS_STATUS_INVALID_VALUE);
  const int ne = static_cast<int>(zr.size());
  const auto clear = [](std::size_t) -> Status { return wwr::WWRBLAS_STATUS_SUCCESS; };
  const auto update = [&](const int step, const T *v_cur, const T *beta_cur,
                          const device::CocgPairs<T> &pairs) {
    device::cocg_update_accumulate<T>(stream, n, k, ne, step, v_cur, beta_cur, pairs, s.p_re,
                                      s.p_im, d_cr, d_ci, d_out);
  };
  return detail::shifted_cocg_run<T>(stream, op, n, k, zr, zi, d_B, s, info, options, clear,
                                     update);
}

} // namespace calaman

namespace calaman {
static_assert(slices_for<ShiftedCocgSlices<float>, int, int, int>);
static_assert(slices_for<ShiftedCocgSlices<double>, int, int, int>);
} // namespace calaman
