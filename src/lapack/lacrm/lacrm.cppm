/**
 * @file lacrm.cppm
 * @brief calaman.lacrm -- C := A * B for a complex A and a real square B,
 *        LAPACK's ?lacrm
 *
 * A is complex M-by-N, B is real N-by-N, C is complex M-by-N; complex only
 * (clacrm / zlacrm), there is no real variant. The reference's mechanism,
 * ported: deinterleave A into a real and an imaginary plane, run one real gemm
 * per plane against B, reassemble C. Here that is one split pass, two
 * wwr::gemm calls and one merge pass, through calaman.complex_cast's strided
 * split_planes / merge_planes -- no device code of its own.
 *
 * Divergence -- workspace is 4*M*N reals, not ZLACRM's 2*M*N: both input planes
 * and both result planes live at once so A is read once and C written once.
 * The planes have leading dimension M, as in the reference.
 *
 * Requires the handle's DEFAULT (host) pointer mode: the gemm scalars are host
 * addresses, as in calaman.larfb. Enqueued on the handle's stream; returns
 * without synchronizing and allocates nothing.
 *
 * | ZLACRM          | calaman::lacrm                               |
 * |-----------------|----------------------------------------------|
 * | c/z variants    | one template over T (wwr complex types)      |
 * | RWORK (2*M*N)   | one device byte buffer + lacrm_bufferSize    |
 * | no XERBLA       | invalid-value Status on bad dims / pointers  |
 *
 * Usage:
 *   import calaman.lacrm;    // also re-exports calaman::Status
 *   const std::size_t bytes = calaman::lacrm_bufferSize<wwr::wwrDoubleComplex>(m, n);
 *   // d_work: device buffer of `bytes`; d_a complex m x n, d_b real n x n
 *   calaman::lacrm<wwr::wwrDoubleComplex>(blas, m, n, d_a, lda, d_b, ldb, d_c, ldc,
 *                                         d_work, bytes);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment;
// needs calaman::Status visible at expansion (the export import below).
#include "error_handling/error_macros.h"

export module calaman.lacrm;

import std;
import wwr.blas;             // wwrblasHandle_t, wwrblasGetStream, WWRBLAS_OP_N, statuses
import wwr.runtime_api;      // wwrStream_t, wwrGetLastError, wwrSuccess
import wwr.complex;          // wwrFloatComplex, wwrDoubleComplex
import wwr.wrappers.blas;    // gemm
import calaman.common;       // complex_fp, ComplexToRealType, WorkspaceLayout, carve_workspace
import calaman.complex_cast; // split_planes, merge_planes

// export import: lacrm RETURNS calaman::Status, so a consumer must see its
// members, not just its name -- the re-export larfb does.
export import calaman.error_handling;

namespace calaman {

namespace lacrm_detail {

/// @brief lacrm()'s workspace: the two planes of A and the two of C, each
///        m-by-n with leading dimension m
template<typename R>
struct LacrmSlices {
  R *a_re = nullptr;
  R *a_im = nullptr;
  R *c_re = nullptr;
  R *c_im = nullptr;

  /// @brief The ONLY description of the layout, run for sizing and carving alike
  void carve(WorkspaceLayout &layout, const int m, const int n) {
    const std::size_t plane = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
    a_re = layout.fixed<R>(plane);
    a_im = layout.fixed<R>(plane);
    c_re = layout.fixed<R>(plane);
    c_im = layout.fixed<R>(plane);
  }
};

static_assert(slices_for<LacrmSlices<double>, int, int>);

} // namespace lacrm_detail

/// @brief Device workspace lacrm() needs, in bytes: four m-by-n real planes
///
/// @tparam T Complex element type (wwrFloatComplex or wwrDoubleComplex)
/// @param m  Row count of A and C
/// @param n  Column count of A and C, order of B
export template<complex_fp T>
std::size_t lacrm_bufferSize(const int m, const int n) {
  if (m <= 0 || n <= 0) {
    return 0;
  }
  return carve_workspace<lacrm_detail::LacrmSlices<ComplexToRealType<T>>>(nullptr, nullptr, m, n);
}

/// @brief C := A * B, A complex m-by-n, B real n-by-n, C complex m-by-n (?lacrm)
///
/// Quick return (nothing enqueued, C untouched) when m or n is 0. Returns
/// invalid-value for negative dimensions, a leading dimension below its
/// minimum, a null pointer, or an undersized @p d_work.
///
/// @tparam T        Complex element type (wwrFloatComplex or wwrDoubleComplex)
/// @param handle    BLAS handle in host pointer mode; its stream carries every op
/// @param m         Row count of A and C
/// @param n         Column count of A and C, order of B
/// @param d_a       Device complex A, column-major, lda >= max(1, m)
/// @param lda       Leading dimension of A
/// @param d_b       Device real B, column-major, ldb >= max(1, n)
/// @param ldb       Leading dimension of B
/// @param d_c       Device complex C, column-major, ldc >= max(1, m); overwritten
/// @param ldc       Leading dimension of C
/// @param d_work    Device workspace of at least lacrm_bufferSize<T>(m, n) bytes
/// @param work_bytes Size of @p d_work in bytes
/// @return Success, invalid-value, a gemm status, or a launch error
export template<complex_fp T>
Status lacrm(const wwr::wwrblasHandle_t handle, const int m, const int n, const T *const d_a,
             const int lda, const ComplexToRealType<T> *const d_b, const int ldb, T *const d_c,
             const int ldc, void *const d_work, const std::size_t work_bytes) {
  using R = ComplexToRealType<T>;
  if (m < 0 || n < 0 || lda < std::max(1, m) || ldb < std::max(1, n) || ldc < std::max(1, m)) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  if (m == 0 || n == 0) {
    return wwr::wwrSuccess;
  }
  if (d_a == nullptr || d_b == nullptr || d_c == nullptr || d_work == nullptr) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  lacrm_detail::LacrmSlices<R> ws;
  if (work_bytes < carve_workspace(d_work, &ws, m, n)) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(handle, &stream));

  const auto um = static_cast<std::size_t>(m);
  const auto un = static_cast<std::size_t>(n);
  split_planes<T>(stream, um, un, d_a, static_cast<std::size_t>(lda), ws.a_re, ws.a_im, um);
  CLM_TRY(wwr::wwrGetLastError());

  // One real gemm per plane, as ZLACRM's two DGEMMs: C_re = A_re * B, C_im = A_im * B.
  const R one{1};
  const R zero{0};
  CLM_TRY(wwr::gemm<R>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, n, n, &one, ws.a_re, m,
                       d_b, ldb, &zero, ws.c_re, m));
  CLM_TRY(wwr::gemm<R>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, n, n, &one, ws.a_im, m,
                       d_b, ldb, &zero, ws.c_im, m));

  merge_planes<T>(stream, um, un, ws.c_re, ws.c_im, um, d_c, static_cast<std::size_t>(ldc));
  return wwr::wwrGetLastError();
}

} // namespace calaman
