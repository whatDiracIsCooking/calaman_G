/**
 * @file larfx.cppm
 * @brief The calaman.larfx module -- apply an elementary reflector, fused for
 *        small orders
 *
 * One host routine that applies the elementary (Householder) reflector
 * H = I - tau * v * v^T to a matrix C in place, for side L (C := H*C) or side R
 * (C := C*H) -- the same result as calaman.larf, but following LAPACK's ?larfx,
 * which SPLITS on the reflector's order (m for side L, n for side R):
 *
 *   order <  11:  a single fused device kernel, NO workspace. Each independent
 *                 vector of C (a column for side L, a row for side R) is its own
 *                 thread, which forms sum = v^T x and updates x -= tau*sum*v.
 *                 This is ?larfx's inline code -- LAPACK hand-unrolls ten bodies
 *                 for the CPU's registers; the GPU keeps the one thread per
 *                 vector and lets the short v stay in registers, so one
 *                 parametrised reduction replaces all ten (see larfx.cu).
 *   order >= 11:  defer to calaman.larf (one gemv + one ger), which is exactly
 *                 what reference ?larfx does (it calls ?larf with incv = 1). This
 *                 path, and ONLY this path, uses the caller's workspace w.
 *
 * So larfx's value over larf is the small-order case the pivoted-QR panel hits
 * constantly: no workspace and a single kernel launch instead of two BLAS-2
 * calls staged through a scratch vector. The two paths agree to rounding; the
 * fused path even accumulates v^T x left-to-right as the reference inline code
 * does.
 *
 * Unlike calaman.larf there is NO incv: ?larfx hardcodes a unit stride on v (its
 * fallback calls ?larf with incv = 1), so v is a plain length-order vector,
 * v(1) stored (not the implicit unit of calaman.larf1f). w is a caller-provided
 * device workspace, length n for side L and m for side R -- referenced only when
 * order >= 11, matching ?larfx's "WORK is not referenced if H has order < 11".
 *
 * tau == 0 makes H the identity: larfx returns immediately, enqueuing nothing
 * and leaving C bitwise-unchanged, exactly as reference ?larfx returns at its
 * top. The empty case (m == 0 or n == 0) enqueues nothing. Requires the handle's
 * DEFAULT (host) pointer mode, inherited wholesale from calaman.larf on the
 * fallback path; the fused path reads only host scalars. v, w and C are device
 * pointers the caller owns; the routine allocates nothing. Templated over float
 * and double.
 *
 * Usage:
 *   import calaman.larfx;    // names calaman::larfx, calaman::Side
 *   import wwr.blas;         // wwrblasHandle_t, wwrblasCreate, ...
 *   // d_C: m x n device matrix, ldc; d_v: reflector (length m for L, n for R);
 *   // d_w: workspace length n (L) or m (R), used only when the order is >= 11
 *   calaman::larfx<double>(handle, calaman::Side::L, m, n, d_v, tau, d_C, ldc,
 *                          d_w);
 */

module;

#include "larfx_bridge.h"
// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the export import
// below supplies.
#include "error_handling/error_macros.h"

export module calaman.larfx;

import std;               // std::size_t
import wwr.blas;          // wwrblasHandle_t, wwrblasStatus_t, wwrblasGetStream, WWRBLAS_STATUS_*
import wwr.runtime_api;   // wwrStream_t, wwrGetLastError, wwrSuccess
import calaman.common;    // Side (:enums)
import calaman.larf;      // calaman::larf -- the order >= 11 fallback

// export import, not a plain import: larfx RETURNS calaman::Status, so a consumer
// of `import calaman.larfx;` must see Status's member functions, not just its
// name -- the same re-export calaman.larf does.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Side (L / R) lives in calaman.common's :enums partition. Re-export it so
// `import calaman.larfx;` alone names calaman::Side, as the usage example relies
// on -- the same re-export calaman.larf does.
export using calaman::Side;

/// @brief Apply H = I - tau*v*v^T to C in place, fused for order < 11
///
/// side L computes C := H*C (reflector order m), side R computes C := C*H (order
/// n). For order < 11 a single workspace-free kernel applies the reflector; for
/// order >= 11 it defers to calaman::larf (gemv + ger), the only path that reads
/// @p w. Short-circuits: a failed launch or BLAS call returns its Status. Does
/// nothing and returns success when @p m or @p n is 0; tau == 0 is a no-op on C.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle GPU BLAS handle in host pointer mode; v, w and C live on its device
/// @param side Side::L for C := H*C, Side::R for C := C*H
/// @param m Row count of C
/// @param n Column count of C
/// @param v Reflector device vector, length m (side L) or n (side R), unit stride
/// @param tau The reflector scalar; tau == 0 leaves C unchanged
/// @param C Device matrix, m by n, column-major, updated in place
/// @param ldc Leading dimension of C (>= m)
/// @param w Device workspace vector, length n (side L) or m (side R), stride 1;
///          referenced only when the reflector order is >= 11
/// @return Success, the launch error, or calaman::larf's Status on the fallback
export template<typename T>
Status larfx(wwr::wwrblasHandle_t handle, const Side side, const int m, const int n, const T *v,
             const T tau, T *C, const int ldc, T *w) {
  if (m == 0 || n == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
  // tau == 0 => H = I. Return before touching the stream so C is bitwise
  // untouched, exactly as reference ?larfx returns at its top.
  if (tau == T{0}) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  // The reflector's order is the dimension it spans: m for side L, n for side R.
  const int order = side == Side::L ? m : n;

  // order >= 11: reference ?larfx defers to ?larf (incv = 1). calaman::larf is
  // that gemv + ger, and the only path that uses the caller's workspace w.
  if (order >= 11) {
    return larf<T>(handle, side, m, n, v, 1, tau, C, ldc, w);
  }

  // order < 11: the fused, workspace-free kernel. side L walks C's columns
  // (count n, order m, within 1, between ldc); side R walks its rows (count m,
  // order n, within ldc, between 1). The kernel needs the handle's stream.
  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(handle, &stream));

  const bool left = side == Side::L;
  const std::size_t count = static_cast<std::size_t>(left ? n : m);
  const std::size_t within = left ? std::size_t{1} : static_cast<std::size_t>(ldc);
  const std::size_t between = left ? static_cast<std::size_t>(ldc) : std::size_t{1};
  device::larfx<T>(stream, count, static_cast<std::size_t>(order), within, between, v, tau, C);
  // The launcher returns void, so a bad launch surfaces only through the
  // runtime's sticky error -- checked here the moment it is enqueued, as lascl2
  // does.
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
