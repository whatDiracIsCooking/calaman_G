/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lacn2 -- estimate the 1-norm of a square
 *        matrix by reverse communication, LAPACK's ?lacn2 (Hager / Higham)
 *
 * The caller never hands over A: each call returns with @p kase set, and the
 * caller overwrites @p d_x with A*x (kase 1) or A^T*x (kase 2) and calls again,
 * until kase comes back 0 with @p est holding a lower bound on ||A||_1 and
 * @p d_v = A*w for the w that attains it. The state is the reference's: kase,
 * isave (indices 1-based, as in Fortran) and est on the host, ISGN in the
 * workspace -- so the same workspace must be passed on every call of one run.
 *
 * Host-driven like the reference: the ?asum / i?amax reductions run through the
 * BLAS handle, which must be in DEFAULT (host) pointer mode, and each call
 * synchronizes the handle's stream once or twice to read them. The elementwise
 * passes are lacn2.cu's kernels. Real only (s/d), as ?lacn2 is.
 *
 * Usage:
 *   import calaman.lacn2;   // also re-exports calaman::Status
 *   const std::size_t bytes = calaman::lacn2_bufferSize<double>(n);
 *   double est = 0; int kase = 0; std::array<int, 3> isave{};
 *   do {
 *     calaman::lacn2<double>(blas, n, d_v, d_x, d_work, bytes, est, kase, isave);
 *     if (kase == 1) { apply A to d_x in place; } else if (kase == 2) { apply A^T; }
 *   } while (kase != 0);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment.
#include "error_handling/error_macros.h"

#include "lacn2_bridge.h"

export module calaman.lacn2;

import std;
import wwr.blas;          // wwrblasHandle_t, wwrblasGetStream, status codes
import wwr.runtime_api;   // wwrMemcpyAsync, wwrMemsetAsync, wwrStreamSynchronize
import wwr.wrappers.blas; // asum, iamax, copy
import calaman.common;    // real_fp, WorkspaceLayout, carve_workspace

export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

namespace lacn2_detail {

/// @brief lacn2()'s workspace: ISGN plus the device flag of the sign test
struct Lacn2Slices {
  int *isgn = nullptr; ///< the previous sign vector, live across calls
  int *flag = nullptr; ///< 1 when the new sign vector differs from isgn

  void carve(WorkspaceLayout &layout, const int len) {
    isgn = layout.fixed<int>(static_cast<std::size_t>(len < 1 ? 1 : len));
    flag = layout.fixed<int>(1);
  }
};

static_assert(slices_for<Lacn2Slices, int>);

/// @brief Copy one device value to the host, synchronizing @p stream
template<typename V>
Status read1(const wwr::wwrStream_t stream, const V *const p, V &out) {
  CLM_TRY(wwr::wwrMemcpyAsync(&out, p, sizeof(V), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace lacn2_detail

/// @brief Device workspace lacn2() needs for order @p n, in bytes
export template<real_fp T>
std::size_t lacn2_bufferSize(const int n) {
  return carve_workspace<lacn2_detail::Lacn2Slices>(nullptr, nullptr, n);
}

/// @brief One reverse-communication step of the 1-norm estimator (LAPACK ?lacn2)
///
/// Start with @p kase = 0; on each return with kase 1 / 2, overwrite @p d_x
/// with A*x / A^T*x and call again with everything else unchanged. kase 0 on
/// return means done: @p est <= ||A||_1, and @p d_v = A*w with est = ||v||/||w||.
///
/// @tparam T Element type, float or double
/// @param handle BLAS handle in host pointer mode; its stream carries every op
/// @param n Order of A (>= 1)
/// @param d_v Device vector, length n; A*w on the final return
/// @param d_x Device vector, length n; the vector the caller multiplies
/// @param d_work Device workspace of >= lacn2_bufferSize<T>(n) bytes, kept across calls
/// @param work_bytes Size of @p d_work in bytes
/// @param est Host estimate; carried between calls
/// @param kase Host state: 0 to start / done, 1 for A*x, 2 for A^T*x
/// @param isave Host state carried between calls (1-based, as the reference)
/// @return Success, InvalidValue for bad arguments or state, or the failing step
export template<real_fp T>
Status lacn2(wwr::wwrblasHandle_t handle, const int n, T *const d_v, T *const d_x,
             void *const d_work, const std::size_t work_bytes, T &est, int &kase,
             std::array<int, 3> &isave) {
  using lacn2_detail::read1;
  constexpr int kItmax = 5;
  const Status kOk = wwr::WWRBLAS_STATUS_SUCCESS;

  if (n < 1 || d_v == nullptr || d_x == nullptr || d_work == nullptr ||
      (kase != 0 && (isave[0] < 1 || isave[0] > 5))) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  lacn2_detail::Lacn2Slices ws;
  if (work_bytes < carve_workspace(d_work, &ws, n)) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(handle, &stream));

  // Label 50: x = e_j for the current column guess, then ask for A*x.
  auto to_unit = [&]() -> Status {
    device::lacn2_unit<T>(stream, n, isave[1] - 1, d_x);
    CLM_TRY(wwr::wwrGetLastError());
    kase = 1;
    isave[0] = 3;
    return kOk;
  };
  // Label 120: the alternating-sign test vector, then ask for A*x.
  auto to_final = [&]() -> Status {
    device::lacn2_altsgn<T>(stream, n, d_x);
    CLM_TRY(wwr::wwrGetLastError());
    kase = 1;
    isave[0] = 5;
    return kOk;
  };
  // x = sign(x), recorded in ISGN, then ask for A^T*x.
  auto to_sign = [&](const int next) -> Status {
    device::lacn2_sign<T>(stream, n, d_x, ws.isgn);
    CLM_TRY(wwr::wwrGetLastError());
    kase = 2;
    isave[0] = next;
    return kOk;
  };

  if (kase == 0) {
    device::lacn2_fill<T>(stream, n, T{1} / static_cast<T>(n), d_x);
    CLM_TRY(wwr::wwrGetLastError());
    kase = 1;
    isave[0] = 1;
    return kOk;
  }

  switch (isave[0]) {
  case 1: { // Label 20: x holds A*x for x = e/n.
    if (n == 1) {
      CLM_TRY(wwr::copy<T, int>(handle, n, d_x, 1, d_v, 1));
      T x0{};
      CLM_TRY(read1(stream, d_x, x0));
      est = std::abs(x0);
      kase = 0;
      return kOk;
    }
    CLM_TRY(wwr::asum<T, int>(handle, n, d_x, 1, &est));
    return to_sign(2);
  }
  case 2: { // Label 40: x holds A^T*sign; first column guess.
    CLM_TRY(wwr::iamax<T, int>(handle, n, d_x, 1, &isave[1]));
    isave[2] = 2;
    return to_unit();
  }
  case 3: { // Label 70: x holds A*e_j.
    CLM_TRY(wwr::copy<T, int>(handle, n, d_x, 1, d_v, 1));
    const T estold = est;
    CLM_TRY(wwr::asum<T, int>(handle, n, d_v, 1, &est));
    CLM_TRY(wwr::wwrMemsetAsync(ws.flag, 0, sizeof(int), stream));
    device::lacn2_sign_changed<T>(stream, n, d_x, ws.isgn, ws.flag);
    CLM_TRY(wwr::wwrGetLastError());
    int changed = 0;
    CLM_TRY(read1(stream, ws.flag, changed));
    // A repeated sign vector means convergence; a non-increasing est, cycling.
    if (changed == 0 || est <= estold) {
      return to_final();
    }
    return to_sign(4);
  }
  case 4: { // Label 110: x holds A^T*sign; next column guess.
    const int jlast = isave[1];
    CLM_TRY(wwr::iamax<T, int>(handle, n, d_x, 1, &isave[1]));
    T xlast{};
    T xnew{};
    CLM_TRY(read1(stream, d_x + (jlast - 1), xlast));
    CLM_TRY(read1(stream, d_x + (isave[1] - 1), xnew));
    if (xlast != std::abs(xnew) && isave[2] < kItmax) {
      ++isave[2];
      return to_unit();
    }
    return to_final();
  }
  default: { // Label 140: x holds A*altsgn.
    T sum{};
    CLM_TRY(wwr::asum<T, int>(handle, n, d_x, 1, &sum));
    const T temp = T{2} * (sum / static_cast<T>(3 * n));
    if (temp > est) {
      CLM_TRY(wwr::copy<T, int>(handle, n, d_x, 1, d_v, 1));
      est = temp;
    }
    kase = 0;
    return kOk;
  }
  }
}

extern template std::size_t lacn2_bufferSize<float>(int);
extern template std::size_t lacn2_bufferSize<double>(int);
extern template Status lacn2<float>(wwr::wwrblasHandle_t, int, float *, float *, void *,
                                    std::size_t, float &, int &, std::array<int, 3> &);
extern template Status lacn2<double>(wwr::wwrblasHandle_t, int, double *, double *, void *,
                                     std::size_t, double &, int &, std::array<int, 3> &);

} // namespace calaman
