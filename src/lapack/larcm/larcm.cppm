/**
 * @file larcm.cppm
 * @brief calaman.larcm -- C := A * B for a real square A and a complex B,
 *        LAPACK's ?larcm
 *
 * A is real M-by-M, B is complex M-by-N, C is complex M-by-N; complex only
 * (clarcm / zlarcm), there is no real variant. The mirror of calaman.lacrm:
 * here the SECOND operand is split -- deinterleave B into a real and an
 * imaginary plane, run one real gemm per plane with A on the left, reassemble
 * C, through calaman.complex_cast's split_planes / merge_planes. No device
 * code of its own.
 *
 * Divergence -- workspace is 4*M*N reals, not ZLARCM's 2*M*N: both input planes
 * and both result planes live at once so B is read once and C written once.
 * The planes have leading dimension M, as in the reference.
 *
 * Requires the handle's DEFAULT (host) pointer mode: the gemm scalars are host
 * addresses, as in calaman.lacrm. Enqueued on the handle's stream; returns
 * without synchronizing and allocates nothing.
 *
 * Usage:
 *   import calaman.larcm;    // also re-exports calaman::Status
 *   const std::size_t bytes = calaman::larcm_bufferSize<wwr::wwrDoubleComplex>(m, n);
 *   // d_work: device buffer of `bytes`; d_a real m x m, d_b complex m x n
 *   calaman::larcm<wwr::wwrDoubleComplex>(blas, m, n, d_a, lda, d_b, ldb, d_c, ldc,
 *                                         d_work, bytes);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment;
// needs calaman::Status visible at expansion (the export import below).
#include "error_handling/error_macros.h"

export module calaman.larcm;

import std;
import wwr.blas;             // wwrblasHandle_t, wwrblasGetStream, WWRBLAS_OP_N, statuses
import wwr.runtime_api;      // wwrStream_t, wwrGetLastError, wwrSuccess
import wwr.complex;          // wwrFloatComplex, wwrDoubleComplex
import wwr.wrappers.blas;    // gemm
import calaman.common;       // complex_fp, ComplexToRealType, WorkspaceLayout, carve_workspace
import calaman.complex_cast; // split_planes, merge_planes

// export import: larcm RETURNS calaman::Status, so a consumer must see its
// members, not just its name.
export import calaman.error_handling;

namespace calaman {

namespace larcm_detail {

/// @brief larcm()'s workspace: the two planes of B and the two of C, each
///        m-by-n with leading dimension m
template<typename R>
struct LarcmSlices {
  R *b_re = nullptr;
  R *b_im = nullptr;
  R *c_re = nullptr;
  R *c_im = nullptr;

  /// @brief The ONLY description of the layout, run for sizing and carving alike
  void carve(WorkspaceLayout &layout, const int m, const int n) {
    const std::size_t plane = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
    b_re = layout.fixed<R>(plane);
    b_im = layout.fixed<R>(plane);
    c_re = layout.fixed<R>(plane);
    c_im = layout.fixed<R>(plane);
  }
};

static_assert(slices_for<LarcmSlices<double>, int, int>);

} // namespace larcm_detail

/// @brief Device workspace larcm() needs, in bytes: four m-by-n real planes
///
/// @tparam T Complex element type (wwrFloatComplex or wwrDoubleComplex)
/// @param m  Order of A, row count of B and C
/// @param n  Column count of B and C
export template<complex_fp T>
std::size_t larcm_bufferSize(const int m, const int n) {
  if (m <= 0 || n <= 0) {
    return 0;
  }
  return carve_workspace<larcm_detail::LarcmSlices<ComplexToRealType<T>>>(nullptr, nullptr, m, n);
}

/// @brief C := A * B, A real m-by-m, B complex m-by-n, C complex m-by-n (?larcm)
///
/// Quick return (nothing enqueued, C untouched) when m or n is 0. Returns
/// invalid-value for negative dimensions, a leading dimension below its
/// minimum, a null pointer, or an undersized @p d_work.
///
/// @tparam T        Complex element type (wwrFloatComplex or wwrDoubleComplex)
/// @param handle    BLAS handle in host pointer mode; its stream carries every op
/// @param m         Order of A, row count of B and C
/// @param n         Column count of B and C
/// @param d_a       Device real A, column-major, lda >= max(1, m)
/// @param lda       Leading dimension of A
/// @param d_b       Device complex B, column-major, ldb >= max(1, m)
/// @param ldb       Leading dimension of B
/// @param d_c       Device complex C, column-major, ldc >= max(1, m); overwritten
/// @param ldc       Leading dimension of C
/// @param d_work    Device workspace of at least larcm_bufferSize<T>(m, n) bytes
/// @param work_bytes Size of @p d_work in bytes
/// @return Success, invalid-value, a gemm status, or a launch error
export template<complex_fp T>
Status larcm(const wwr::wwrblasHandle_t handle, const int m, const int n,
             const ComplexToRealType<T> *const d_a, const int lda, const T *const d_b,
             const int ldb, T *const d_c, const int ldc, void *const d_work,
             const std::size_t work_bytes) {
  using R = ComplexToRealType<T>;
  if (m < 0 || n < 0 || lda < std::max(1, m) || ldb < std::max(1, m) || ldc < std::max(1, m)) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  if (m == 0 || n == 0) {
    return wwr::wwrSuccess;
  }
  if (d_a == nullptr || d_b == nullptr || d_c == nullptr || d_work == nullptr) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  larcm_detail::LarcmSlices<R> ws;
  if (work_bytes < carve_workspace(d_work, &ws, m, n)) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(handle, &stream));

  const auto um = static_cast<std::size_t>(m);
  const auto un = static_cast<std::size_t>(n);
  split_planes<T>(stream, um, un, d_b, static_cast<std::size_t>(ldb), ws.b_re, ws.b_im, um);
  CLM_TRY(wwr::wwrGetLastError());

  // One real gemm per plane, as ZLARCM's two DGEMMs: C_re = A * B_re, C_im = A * B_im.
  const R one{1};
  const R zero{0};
  CLM_TRY(wwr::gemm<R>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, n, m, &one, d_a, lda,
                       ws.b_re, m, &zero, ws.c_re, m));
  CLM_TRY(wwr::gemm<R>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, n, m, &one, d_a, lda,
                       ws.b_im, m, &zero, ws.c_im, m));

  merge_planes<T>(stream, um, un, ws.c_re, ws.c_im, um, d_c, static_cast<std::size_t>(ldc));
  return wwr::wwrGetLastError();
}

} // namespace calaman
