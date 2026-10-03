/**
 * @file interface.cppm
 * @brief Primary interface for calaman.orghr -- generate the orthogonal Q from
 *        the Householder reflectors ?gehrd produced
 *
 * The GPU counterpart of LAPACK's ?orghr, real only (float/double). ?gehrd
 * reduces A to upper Hessenberg form by Q^T A Q = H and leaves Q as a product of
 * ihi-ilo elementary reflectors packed below the subdiagonal of the window
 * columns [ilo, ihi-1], with their scalars in tau[ilo-1 .. ihi-2]. orghr forms
 * the explicit n-by-n Q from them.
 *
 * It is NOT a reimplementation of ?orgqr: it is the ?orghr-specific setup around
 * one ?orgqr call. The reflectors sit one column too far left to be a QR
 * factorization of the window block, so the device shuffle (orghr.cu) shifts
 * them one column right, fills the diagonal with ones and the leading ilo /
 * trailing n-ihi rows and columns with the identity, then this hands the nh-by-nh
 * window block A(ilo+1:ihi, ilo+1:ihi) to wwr::orgqr with its nh reflectors. The
 * shuffle writes into a device snapshot so the right-to-left column move is a
 * race-free gather rather than an aliasing in-place shift -- the snapshot lives
 * in the caller's workspace, sized by orghr_bufferSize.
 *
 * Mapping from LAPACK's DORGHR (docs/architecture.md §4 -- keep the name, drop
 * the Fortran calling convention):
 *
 * | DORGHR            | calaman::orghr                          |
 * |-------------------|-----------------------------------------|
 * | s/d variants      | one template over T (float, double)     |
 * | INTEGER ilo/ihi/n | kept as int, 1-based, as LAPACK         |
 * | WORK / LWORK      | one device byte buffer + orghr_bufferSize |
 * | INFO              | wwrsolverStatus_t + the ?orgqr devInfo  |
 *
 * `extern template` below pairs with instantiations.cpp: the wrapper is
 * instantiated once inside this library, so an importer never re-instantiates a
 * body that names the .cu-side launcher declared only in the GMF.
 *
 * Usage:
 *   import calaman.orghr;        // also re-exports calaman::Status
 *   import wwr.solver;           // wwrsolverDnHandle_t, wwrsolverDnCreate
 *   wwr::wwrsolverDnHandle_t solver{};
 *   wwr::wwrsolverDnCreate(&solver);
 *   wwr::wwrsolverDnSetStream(solver, stream);
 *   const std::size_t bytes = calaman::orghr_bufferSize<T>(solver, n, lda, ilo, ihi);
 *   // d_work: device buffer of `bytes`; d_a: packed reflectors from ?gehrd
 *   calaman::orghr<T>(solver, n, ilo, ihi, d_a, lda, d_tau, d_work, bytes, d_info);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

#include "orghr_bridge.h"

export module calaman.orghr;

import std;
import wwr.blas;            // WWRBLAS_STATUS_INVALID_VALUE / _INTERNAL_ERROR (the validation codes)
import wwr.runtime_api;     // wwrStream_t, wwrMemcpy/MemsetAsync, wwrGetLastError, wwrSuccess
import wwr.solver;          // wwrsolverDnHandle_t, WWRSOLVER_STATUS_SUCCESS, wwrsolverDnGetStream
import wwr.wrappers.solver; // orgqr, orgqr_bufferSize
import calaman.common;      // WorkspaceLayout

// export import, not a plain import: orghr RETURNS calaman::Status, so a consumer
// of `import calaman.orghr;` must see Status's member functions, not just its
// name -- the same re-export laset does.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

namespace orghr_detail {

/// @brief Query ?orgqr's workspace length (T elements) for the nh-by-nh window
///
/// nh = ihi - ilo is the number of reflectors ?orgqr regenerates. Returns 1 on an
/// empty window (nh < 1) or any query failure, so the layout always reserves at
/// least one element.
template<typename T>
int orgqr_work_len(wwr::wwrsolverDnHandle_t solver, const int n, const int ilo, const int ihi) {
  const int nh = ihi - ilo;
  if (nh < 1) {
    return 1;
  }
  const T *dummy = nullptr;
  int lwork = 0;
  if (wwr::orgqr_bufferSize<T>(solver, nh, nh, nh, dummy, n, dummy, &lwork) !=
      wwr::WWRSOLVER_STATUS_SUCCESS) {
    return 1;
  }
  return lwork < 1 ? 1 : lwork;
}

/// @brief Carve orghr()'s workspace, SIZING over @p base == nullptr or carving it
///
/// Two fixed regions: the n-by-n snapshot the shuffle gathers from, and ?orgqr's
/// own scratch (queried at the nh window). Run once with a null base inside
/// orghr_bufferSize and once over the real buffer inside orghr, so the size can
/// never drift from the carving (see WorkspaceLayout).
template<typename T>
struct OrghrWorkspace {
  T *snapshot;  ///< untouched copy of the input, the shuffle's gather source
  T *orgqr_work; ///< ?orgqr's scratch on the window block
  int orgqr_lwork;
  std::size_t total; ///< bytes the layout spans
};

template<typename T>
OrghrWorkspace<T> map_workspace(wwr::wwrsolverDnHandle_t solver, void *base, const int n,
                                const int lda, const int ilo, const int ihi) {
  // The snapshot spans the full lda-by-n input the caller passes, since the copy
  // is lda-strided; lda >= max(1, n) is already validated at the call sites.
  const std::size_t span = static_cast<std::size_t>(lda < 1 ? 1 : lda) *
                           static_cast<std::size_t>(n < 1 ? 1 : n);
  const int lwork = orgqr_work_len<T>(solver, n, ilo, ihi);

  WorkspaceLayout layout(base);
  OrghrWorkspace<T> ws{};
  ws.snapshot    = layout.fixed<T>(span);
  ws.orgqr_work  = layout.fixed<T>(static_cast<std::size_t>(lwork));
  ws.orgqr_lwork = lwork;
  ws.total       = layout.total();
  return ws;
}

} // namespace orghr_detail

// ========================================================================
// Workspace size (exported)
// ========================================================================

/// @brief Device workspace orghr() needs, in bytes
///
/// Covers the lda-by-n snapshot the reflector shuffle gathers from plus ?orgqr's
/// own scratch on the nh = ihi-ilo window. @p ilo / @p ihi are 1-based; pass the
/// same @p lda given to orghr().
///
/// @tparam T     Element type (float or double)
/// @param solver Solver handle, used only to size ?orgqr's workspace
/// @param n      Order of Q
/// @param lda    Leading dimension of A (lda >= max(1, n))
/// @param ilo    First window index (1-based)
/// @param ihi    Last window index (1-based)
export template<typename T>
std::size_t orghr_bufferSize(wwr::wwrsolverDnHandle_t solver, const int n, const int lda,
                             const int ilo, const int ihi) {
  return orghr_detail::map_workspace<T>(solver, nullptr, n, lda, ilo, ihi).total;
}

// ========================================================================
// Driver (exported)
// ========================================================================

/// @brief Generate the n-by-n orthogonal Q from ?gehrd's reflectors, in place
///
/// On entry @p d_a holds the packed ?gehrd output (reflectors below the
/// subdiagonal of columns [ilo, ihi-1]); on exit it holds Q. @p d_tau holds the
/// ihi-ilo reflector scalars in [ilo-1 .. ihi-2]. @p d_work is a device buffer of
/// at least orghr_bufferSize<T>(@p solver, n, ilo, ihi) bytes. @p d_info receives
/// ?orgqr's devInfo (0 on success). @p ilo / @p ihi are 1-based, matching ?gehrd.
///
/// Returns WWRSOLVER_STATUS_SUCCESS on success; WWRSOLVER_STATUS_INVALID_VALUE for
/// bad dimensions, null pointers, or an undersized buffer; otherwise the failing
/// ?orgqr status. A kernel-launch failure surfaces as the runtime error the
/// launch reported, carried through calaman::Status.
///
/// @tparam T      Element type (float or double)
/// @param solver  Solver handle; its stream carries every op
/// @param n       Order of Q
/// @param ilo     First window index (1-based)
/// @param ihi     Last window index (1-based)
/// @param d_a     Device matrix, column-major, leading dimension @p lda; in: ?gehrd
///                reflectors, out: Q
/// @param lda     Leading dimension of A; lda >= max(1, n)
/// @param d_tau   Device reflector scalars (length >= ihi-1)
/// @param d_work  Device workspace, >= orghr_bufferSize bytes
/// @param work_bytes Size of @p d_work in bytes
/// @param d_info  Device int; ?orgqr's devInfo (0 on success)
/// @return Success, invalid-value, the ?orgqr status, or a launch error
export template<typename T>
Status orghr(wwr::wwrsolverDnHandle_t solver, const int n, const int ilo, const int ihi, T *d_a,
             const int lda, const T *d_tau, void *d_work, const std::size_t work_bytes,
             int *d_info) {
  // LAPACK's DORGHR argument validation (INFO < 0), mapped to invalid-value.
  if (n < 0 || (n > 0 && (ilo < 1 || ilo > n)) || ihi < std::min(ilo, n) || ihi > n ||
      lda < std::max(1, n)) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  if (n == 0) {
    return wwr::WWRSOLVER_STATUS_SUCCESS;
  }
  if (d_a == nullptr || d_tau == nullptr || d_work == nullptr || d_info == nullptr) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  const auto ws = orghr_detail::map_workspace<T>(solver, d_work, n, lda, ilo, ihi);
  if (work_bytes < ws.total) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  wwr::wwrStream_t stream{};
  if (wwr::wwrsolverDnGetStream(solver, &stream) != wwr::WWRSOLVER_STATUS_SUCCESS) {
    return wwr::WWRBLAS_STATUS_INTERNAL_ERROR;
  }

  // Snapshot the whole n-by-n input so the shuffle can gather from it while it
  // overwrites d_a -- the right-to-left column move aliases in place, a gather
  // from an untouched copy does not. lda-strided, so copy the full lda*n span.
  const std::size_t span_bytes =
      static_cast<std::size_t>(lda) * static_cast<std::size_t>(n) * sizeof(T);
  CLM_TRY(wwr::wwrMemcpyAsync(ws.snapshot, d_a, span_bytes, wwr::wwrMemcpyDeviceToDevice, stream));

  // Shuffle the reflectors one column right and identity-pad, snapshot -> d_a;
  // catch a bad launch through the runtime's sticky error.
  device::orghr_prep<T>(stream, static_cast<std::size_t>(n), ilo, ihi, ws.snapshot, d_a,
                        static_cast<std::size_t>(lda));
  CLM_TRY(wwr::wwrGetLastError());

  // Regenerate Q on the nh-by-nh window block with ?orgqr; nh reflectors from
  // tau[ilo-1 ..]. An empty window (nh < 1) leaves the identity-padded d_a as Q
  // and no ?orgqr runs, so write the INFO = 0 ?orgqr would have, keeping the
  // quick return's info faithful to LAPACK's success contract.
  const int nh = ihi - ilo;
  if (nh <= 0) {
    CLM_TRY(wwr::wwrMemsetAsync(d_info, 0, sizeof(int), stream));
    return wwr::wwrSuccess;
  }
  {
    const std::size_t off = static_cast<std::size_t>(ilo) + static_cast<std::size_t>(ilo) *
                                                                 static_cast<std::size_t>(lda);
    const auto st = wwr::orgqr<T>(solver, nh, nh, nh, d_a + off, lda, d_tau + (ilo - 1),
                                  ws.orgqr_work, ws.orgqr_lwork, d_info);
    if (st != wwr::WWRSOLVER_STATUS_SUCCESS) {
      return st;
    }
  }
  return wwr::wwrSuccess;
}

extern template Status orghr<float>(wwr::wwrsolverDnHandle_t, int, int, int, float *, int,
                                     const float *, void *, std::size_t, int *);
extern template Status orghr<double>(wwr::wwrsolverDnHandle_t, int, int, int, double *, int,
                                      const double *, void *, std::size_t, int *);

} // namespace calaman
