/**
 * @file interface.cppm
 * @brief Primary interface for calaman.diis -- Pulay DIIS extrapolation over a
 *        device-resident ring of (vector, residual) pairs
 *
 * DIIS accelerates a fixed-point iteration by replacing the newest iterate F
 * with F* = sum_i c_i F_i over the recent history, the c_i minimizing
 * |sum_i c_i e_i| subject to sum_i c_i = 1. Each push writes one (F, e) column,
 * updates the residual Gram incrementally, and -- once two pairs are held --
 * solves the bordered system on the device and writes F*. Nothing is read back
 * to the host. Design notes (ring order, incremental Gram): README.md.
 *
 * The host-side cursor is a DiisState; the histories, Gram and coefficients are
 * carved from one caller-owned workspace sized by diis_bufferSize.
 *
 * Usage:
 *   import calaman.diis;
 *
 *   calaman::DiisState state = calaman::make_diis_state(n * n, 8);
 *   const std::size_t bytes = calaman::diis_bufferSize<double>(state);
 *   // d_work: device buffer of `bytes`, kept for the life of `state`
 *   calaman::diis_push_and_extrapolate<double>(blas, stream, state, d_f, d_e, d_out,
 *                                              d_singular, d_work, bytes);
 */

module;

// CLM_TRY / CLM_REQUIRE are macros, so they arrive by #include in the global
// module fragment. diis_bridge.h declares the device launcher.
#include "diis_bridge.h"
#include "error_handling/error_macros.h"

export module calaman.diis;

import std;
import wwr.blas;              // wwrblasHandle_t, WWRBLAS_OP_*, WWRBLAS_STATUS_INVALID_VALUE
import wwr.runtime_api;       // wwrStream_t, wwrMemcpyAsync, wwrMemsetAsync, wwrGetLastError
import wwr.wrappers.blas;     // gemv
import wwr.extension.blas;    // ScopedPointerMode
import calaman.common;        // real_fp, kOne, kZero, all_nonnull, WorkspaceLayout, carve_workspace

// export import: the entry points RETURN calaman::Status.
export import calaman.error_handling;

export namespace calaman {

/// @brief Largest history diis_push_and_extrapolate accepts; bounds the solve
///        kernel's (cap+1)^2 shared-memory system
inline constexpr int kDiisMaxHistory = 64;

/// @brief Host-side cursor of one DIIS subspace: vector length, history cap, and
///        the ring position. Pair it with one workspace for its whole life.
struct DiisState {
  int len = 0;  ///< elements per vector (n*n for one n x n matrix)
  int cap = 1;  ///< maximum (F, e) pairs held
  int size = 0; ///< pairs currently held, 0..cap
  int head = 0; ///< oldest physical column once full; 0 while growing
};

/// @brief A fresh subspace over length-@p len vectors keeping at most
///        @p max_history pairs (clamped to at least 1)
[[nodiscard]] constexpr DiisState make_diis_state(const int len, const int max_history) noexcept {
  return DiisState{len, max_history < 1 ? 1 : max_history, 0, 0};
}

/// @brief Drop the whole subspace; the workspace is kept and refilled before it
///        is read
constexpr void diis_reset(DiisState &state) noexcept {
  state.size = 0;
  state.head = 0;
}

} // namespace calaman

namespace calaman {

namespace diis_detail {

/// @brief The DIIS workspace: both histories, the persistent Gram, the
///        coefficients -- every region fixed, live across pushes
template<typename T>
struct DiisSlices {
  T *fock = nullptr;  ///< len x cap, column-major, lda = len
  T *error = nullptr; ///< len x cap, column-major, lda = len
  T *gram = nullptr;  ///< cap x cap, upper triangle valid, ld = cap
  T *coeff = nullptr; ///< cap

  void carve(WorkspaceLayout &layout, const int vec_len, const int history_cap) {
    const auto len = static_cast<std::size_t>(vec_len);
    const auto cap = static_cast<std::size_t>(history_cap);
    fock = layout.fixed<T>(len * cap);
    error = layout.fixed<T>(len * cap);
    gram = layout.fixed<T>(cap * cap);
    coeff = layout.fixed<T>(cap);
  }
};

static_assert(slices_for<DiisSlices<double>, int, int>);

} // namespace diis_detail

/// @brief Device workspace diis_push_and_extrapolate needs for @p state, in bytes
///
/// Depends only on `state.len` and `state.cap`; 0 when either is out of range.
///
/// @tparam T Element type (float or double)
export template<real_fp T>
std::size_t diis_bufferSize(const DiisState &state) {
  if (state.len < 1 || state.cap < 1 || state.cap > kDiisMaxHistory) {
    return 0;
  }
  return carve_workspace<diis_detail::DiisSlices<T>>(nullptr, nullptr, state.len, state.cap);
}

/// @brief Push (@p cur_fock, @p cur_residual) and, with two or more pairs held,
///        write the DIIS extrapolation F* into @p out_fock
///
/// Copies both vectors into the ring (overwriting the oldest pair once full),
/// then advances @p state. With fewer than two pairs @p out_fock is untouched;
/// a singular subspace writes the plain newest @p cur_fock. Never synchronizes.
/// On an error return @p state is unchanged but the ring may not be: reset it.
///
/// @tparam T Element type (float or double)
/// @param blas BLAS handle bound to @p stream; forced to host pointer mode inside
/// @param stream Stream every copy, gemv and the solve kernel are enqueued on
/// @param state Subspace cursor; advanced on success
/// @param cur_fock Device vector F (state.len elements), copied in
/// @param cur_residual Device residual e for F (state.len elements), copied in
/// @param out_fock Device output (state.len elements); may alias @p cur_fock
/// @param singular_device Optional 1-int device flag: 1 when the solve fell back
///        to the newest F, 0 otherwise (also 0 with fewer than two pairs)
/// @param d_work Workspace of @p work_bytes, the same buffer on every push
/// @param work_bytes At least diis_bufferSize<T>(state)
/// @param pivot_floor Pivot magnitude below which the subspace is singular
/// @return Success, INVALID_VALUE on bad arguments, or the first failing call
export template<real_fp T>
Status diis_push_and_extrapolate(wwr::wwrblasHandle_t blas, wwr::wwrStream_t stream,
                                 DiisState &state, const T *cur_fock, const T *cur_residual,
                                 T *out_fock, int *singular_device, void *d_work,
                                 const std::size_t work_bytes, const T pivot_floor = T{1e-14}) {
  CLM_REQUIRE(state.len >= 1 && state.cap >= 1 && state.cap <= kDiisMaxHistory,
              wwr::WWRBLAS_STATUS_INVALID_VALUE);
  CLM_REQUIRE(state.size >= 0 && state.size <= state.cap && state.head >= 0 &&
                  state.head < state.cap && (state.head == 0 || state.size == state.cap),
              wwr::WWRBLAS_STATUS_INVALID_VALUE);
  CLM_REQUIRE(all_nonnull(cur_fock, cur_residual, out_fock, d_work),
              wwr::WWRBLAS_STATUS_INVALID_VALUE);
  diis_detail::DiisSlices<T> s;
  CLM_REQUIRE(work_bytes >= carve_workspace(d_work, &s, state.len, state.cap),
              wwr::WWRBLAS_STATUS_INVALID_VALUE);

  // Ring push: append while growing, then overwrite the oldest column. The valid
  // columns stay a contiguous prefix [0, size) or the whole buffer -- never a
  // wrapped range -- which the gemvs below rely on.
  int size = state.size;
  int head = state.head;
  int slot = 0;
  if (size < state.cap) {
    slot = size++;
  } else {
    slot = head;
    head = (head + 1) % state.cap;
  }
  const int len = state.len;
  const int cap = state.cap;
  const auto col_offset = static_cast<std::size_t>(slot) * static_cast<std::size_t>(len);
  const std::size_t col_bytes = sizeof(T) * static_cast<std::size_t>(len);
  T *const e_slot = s.error + col_offset;
  CLM_TRY(wwr::wwrMemcpyAsync(s.error + col_offset, cur_residual, col_bytes,
                              wwr::wwrMemcpyDeviceToDevice, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(s.fock + col_offset, cur_fock, col_bytes,
                              wwr::wwrMemcpyDeviceToDevice, stream));

  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode mode{blas, wwr::WWRBLAS_POINTER_MODE_HOST,
                                               PointerModeStatus{&pm_status}};
  CLM_TRY(pm_status);

  // Incremental Gram: only <e_j, e_slot> changed. Every upper-triangle entry
  // [a][c] is rewritten by the push that wrote column max(a, c), and this runs
  // even with one pair held -- so no entry is read stale, including after a reset.
  // Column segment + diagonal, j = 0..slot: gram[j*cap + slot].
  CLM_TRY(wwr::gemv<T, int>(blas, wwr::WWRBLAS_OP_T, len, slot + 1, &kOne<T>, s.error, len, e_slot,
                            1, &kZero<T>, s.gram + slot, cap));
  if (size > slot + 1) {
    // Row segment, j = slot..size-1: gram[slot*cap + j].
    CLM_TRY(wwr::gemv<T, int>(blas, wwr::WWRBLAS_OP_T, len, size - slot, &kOne<T>, e_slot, len,
                              e_slot, 1, &kZero<T>,
                              s.gram + (static_cast<std::size_t>(slot) * cap) + slot, 1));
  }

  if (size >= 2) {
    device::diis_solve<T>(stream, size, cap, slot, s.gram, s.coeff, singular_device, pivot_floor);
    CLM_TRY(wwr::wwrGetLastError());
    // F* = F_hist c over physical columns 0..size-1; the sum is order-invariant,
    // so the ring's rotated column order needs no unrotation.
    CLM_TRY(wwr::gemv<T, int>(blas, wwr::WWRBLAS_OP_N, len, size, &kOne<T>, s.fock, len, s.coeff,
                              1, &kZero<T>, out_fock, 1));
  } else if (singular_device != nullptr) {
    CLM_TRY(wwr::wwrMemsetAsync(singular_device, 0, sizeof(int), stream));
  }

  state.size = size;
  state.head = head;
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

// Instantiated once in instantiations.cpp.
extern template std::size_t diis_bufferSize<float>(const DiisState &);
extern template std::size_t diis_bufferSize<double>(const DiisState &);
extern template Status diis_push_and_extrapolate<float>(wwr::wwrblasHandle_t, wwr::wwrStream_t,
                                                        DiisState &, const float *, const float *,
                                                        float *, int *, void *, std::size_t,
                                                        float);
extern template Status diis_push_and_extrapolate<double>(wwr::wwrblasHandle_t, wwr::wwrStream_t,
                                                         DiisState &, const double *,
                                                         const double *, double *, int *, void *,
                                                         std::size_t, double);

} // namespace calaman
