/**
 * @file interface.cppm
 * @brief Primary interface for calaman.ilalc -- the last non-zero column of
 *        a matrix, LAPACK's ila?lc
 *
 * Writes to a device int the number of leading columns of an m-by-n
 * column-major A that must be considered: 1 + the 0-based column of A's last
 * non-zero, which is ila?lc's 1-based index, and 0 when A is entirely zero or
 * empty. Enqueued on a stream; returns WITHOUT synchronizing. A two-stage reduction (ilalc.cu) whose
 * per-column intermediate lives in the caller's workspace, sized by
 * ilalc_bufferSize.
 *
 * Mapping from ILADLC (docs/architecture.md §4): s/d/c/z become one template
 * over T; the INTEGER result becomes a device int. Divergence: m == 0 yields 0,
 * where the reference reads A(1,N) of an empty matrix.
 *
 * `extern template` pairs with instantiations.cpp, so an importer never
 * re-instantiates a body that names the GMF-declared .cu launcher.
 *
 * Usage:
 *   import calaman.ilalc;     // also re-exports calaman::Status
 *   const std::size_t bytes = calaman::ilalc_bufferSize(m, n);
 *   // d_A: m-by-n device matrix, ld lda; d_last: device int; d_work: bytes
 *   calaman::ilalc<double>(stream, m, n, d_A, lda, d_last, d_work, bytes);
 */

module;

// CLM_REQUIRE -- a macro, so it arrives by #include in the global module
// fragment; needs calaman::Status visible at expansion (the export import below).
#include "error_handling/error_macros.h"

#include "ilalc_bridge.h"

export module calaman.ilalc;

import std;
import wwr.runtime_api; // wwrStream_t, wwrMemsetAsync, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common;  // WorkspaceLayout, carve_workspace, slices_for

export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

namespace ilalc_detail {

/// @brief ilalc()'s workspace: per column j, j + 1 if it holds a non-zero, else 0
struct IlalcSlices {
  int *col_count = nullptr;

  void carve(WorkspaceLayout &layout, const int m, const int n) {
    const bool empty = m <= 0 || n <= 0;
    col_count = layout.fixed<int>(empty ? 0 : static_cast<std::size_t>(n));
  }
};

static_assert(slices_for<IlalcSlices, int, int>);

} // namespace ilalc_detail

/// @brief Device workspace ilalc() needs for an m-by-n A, in bytes (0 if empty)
export std::size_t ilalc_bufferSize(const int m, const int n) {
  return carve_workspace<ilalc_detail::IlalcSlices>(nullptr, nullptr, m, n);
}

/// @brief Count of leading columns of A up to its last non-zero column (ila?lc)
///
/// Writes 0 to @p d_last when A is all zero or @p m / @p n is 0; the workspace
/// may then be null. Only rows [0, m) of each column are read.
///
/// @tparam T Element type; float, double, wwrFloatComplex or wwrDoubleComplex
/// @param stream     Stream the launches are enqueued on; all pointers live on its device
/// @param d_A        Column-major m-by-n device matrix, leading dimension @p lda
/// @param lda        Leading dimension of @p d_A; lda >= max(1, m)
/// @param d_last     Device int receiving the count, in [0, n]
/// @param d_work     Device workspace of >= ilalc_bufferSize(m, n) bytes
/// @param work_bytes Size of @p d_work in bytes
/// @return Success, wwrErrorInvalidValue for a bad argument or an undersized
///         workspace, or the launch error
export template<typename T>
Status ilalc(const wwr::wwrStream_t stream, const int m, const int n, const T *const d_A,
             const int lda, int *const d_last, void *const d_work, const std::size_t work_bytes) {
  CLM_REQUIRE(m >= 0 && n >= 0 && lda >= std::max(1, m) && d_last != nullptr,
              wwr::wwrErrorInvalidValue);
  if (m == 0 || n == 0) {
    return wwr::wwrMemsetAsync(d_last, 0, sizeof(int), stream);
  }
  CLM_REQUIRE(d_A != nullptr && d_work != nullptr, wwr::wwrErrorInvalidValue);
  ilalc_detail::IlalcSlices ws;
  CLM_REQUIRE(work_bytes >= carve_workspace(d_work, &ws, m, n), wwr::wwrErrorInvalidValue);

  device::ilalc<T>(stream, m, n, d_A, lda, d_last, ws.col_count);
  // The launcher returns void; the sticky launch error is the only report.
  return wwr::wwrGetLastError();
}

extern template Status ilalc<float>(wwr::wwrStream_t, int, int, const float *, int, int *, void *,
                                    std::size_t);
extern template Status ilalc<double>(wwr::wwrStream_t, int, int, const double *, int, int *,
                                     void *, std::size_t);
extern template Status ilalc<wwr::wwrFloatComplex>(wwr::wwrStream_t, int, int,
                                                   const wwr::wwrFloatComplex *, int, int *,
                                                   void *, std::size_t);
extern template Status ilalc<wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, int,
                                                    const wwr::wwrDoubleComplex *, int, int *,
                                                    void *, std::size_t);

} // namespace calaman
