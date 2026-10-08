/**
 * @file interface.cppm
 * @brief Primary interface for calaman.ilalr -- the last non-zero row of a
 *        matrix, LAPACK's ila?lr
 *
 * Writes to a device int the number of leading rows of an m-by-n column-major
 * A that must be considered: 1 + the 0-based row of A's last non-zero, which is
 * ila?lr's 1-based index, and 0 when A is entirely zero or empty. Enqueued on a
 * stream; returns WITHOUT synchronizing. A two-stage reduction (ilalr.cu) whose
 * per-column intermediate lives in the caller's workspace, sized by
 * ilalr_bufferSize.
 *
 * Mapping from ILADLR (docs/architecture.md §4): s/d/c/z become one template
 * over T; the INTEGER result becomes a device int. Divergence: n == 0 yields 0,
 * where the reference reads A(M,1) of an empty matrix.
 *
 * `extern template` pairs with instantiations.cpp, so an importer never
 * re-instantiates a body that names the GMF-declared .cu launcher.
 *
 * Usage:
 *   import calaman.ilalr;     // also re-exports calaman::Status
 *   const std::size_t bytes = calaman::ilalr_bufferSize(m, n);
 *   // d_A: m-by-n device matrix, ld lda; d_last: device int; d_work: bytes
 *   calaman::ilalr<double>(stream, m, n, d_A, lda, d_last, d_work, bytes);
 */

module;

// CLM_REQUIRE -- a macro, so it arrives by #include in the global module
// fragment; needs calaman::Status visible at expansion (the export import below).
#include "error_handling/error_macros.h"

#include "ilalr_bridge.h"

export module calaman.ilalr;

import std;
import wwr.runtime_api; // wwrStream_t, wwrMemsetAsync, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common;  // WorkspaceLayout, carve_workspace, slices_for

export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

namespace ilalr_detail {

/// @brief ilalr()'s workspace: one last-non-zero-row count per column
struct IlalrSlices {
  int *col_last = nullptr;

  void carve(WorkspaceLayout &layout, const int m, const int n) {
    const bool empty = m <= 0 || n <= 0;
    col_last = layout.fixed<int>(empty ? 0 : static_cast<std::size_t>(n));
  }
};

static_assert(slices_for<IlalrSlices, int, int>);

} // namespace ilalr_detail

/// @brief Device workspace ilalr() needs for an m-by-n A, in bytes (0 if empty)
export std::size_t ilalr_bufferSize(const int m, const int n) {
  return carve_workspace<ilalr_detail::IlalrSlices>(nullptr, nullptr, m, n);
}

/// @brief Count of leading rows of A up to its last non-zero row (ila?lr)
///
/// Writes 0 to @p d_last when A is all zero or @p m / @p n is 0; the workspace
/// may then be null. Only rows [0, m) of each column are read.
///
/// @tparam T Element type; float, double, wwrFloatComplex or wwrDoubleComplex
/// @param stream     Stream the launches are enqueued on; all pointers live on its device
/// @param d_A        Column-major m-by-n device matrix, leading dimension @p lda
/// @param lda        Leading dimension of @p d_A; lda >= max(1, m)
/// @param d_last     Device int receiving the count, in [0, m]
/// @param d_work     Device workspace of >= ilalr_bufferSize(m, n) bytes
/// @param work_bytes Size of @p d_work in bytes
/// @return Success, wwrErrorInvalidValue for a bad argument or an undersized
///         workspace, or the launch error
export template<typename T>
Status ilalr(const wwr::wwrStream_t stream, const int m, const int n, const T *const d_A,
             const int lda, int *const d_last, void *const d_work, const std::size_t work_bytes) {
  CLM_REQUIRE(m >= 0 && n >= 0 && lda >= std::max(1, m) && d_last != nullptr,
              wwr::wwrErrorInvalidValue);
  if (m == 0 || n == 0) {
    return wwr::wwrMemsetAsync(d_last, 0, sizeof(int), stream);
  }
  CLM_REQUIRE(d_A != nullptr && d_work != nullptr, wwr::wwrErrorInvalidValue);
  ilalr_detail::IlalrSlices ws;
  CLM_REQUIRE(work_bytes >= carve_workspace(d_work, &ws, m, n), wwr::wwrErrorInvalidValue);

  device::ilalr<T>(stream, m, n, d_A, lda, d_last, ws.col_last);
  // The launcher returns void; the sticky launch error is the only report.
  return wwr::wwrGetLastError();
}

extern template Status ilalr<float>(wwr::wwrStream_t, int, int, const float *, int, int *, void *,
                                    std::size_t);
extern template Status ilalr<double>(wwr::wwrStream_t, int, int, const double *, int, int *,
                                     void *, std::size_t);
extern template Status ilalr<wwr::wwrFloatComplex>(wwr::wwrStream_t, int, int,
                                                   const wwr::wwrFloatComplex *, int, int *,
                                                   void *, std::size_t);
extern template Status ilalr<wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, int,
                                                    const wwr::wwrDoubleComplex *, int, int *,
                                                    void *, std::size_t);

} // namespace calaman
