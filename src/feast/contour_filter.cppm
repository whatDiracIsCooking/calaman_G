/**
 * @file contour_filter.cppm
 * @brief The FEAST filter rho(A) Y, as Ne shifted solves on one stream
 *
 * The :contour_filter partition of calaman.feast.
 *
 *   rho(A) Y = sum_e Re[ w_e (Z_e I - A)^{-1} Y ]        (see :compute_quadrature)
 *
 * The Ne shifted systems are independent, and each stage over them is one
 * batched call: getrfBatched factors all Ne resolvents once per solve -- neither
 * Z_e nor A changes between iterations -- and getrsBatched solves all Ne of them
 * against Y once per iteration. Around those, three small kernels build the
 * resolvents, broadcast the real Y into the Ne complex right-hand sides, and
 * reduce the Ne solutions to the real filtered subspace.
 *
 * Z_e I - A is complex symmetric, not Hermitian, so the factorization is a
 * general LU. It is never singular: every Z_e has a positive imaginary part and
 * A's spectrum is real.
 *
 * Both functions expect the BLAS handle's stream to be @p stream. A kernel-launch
 * failure surfaces through wwr::wwrGetLastError(); a batched-BLAS failure through
 * its own status. Pivot breakdowns land in s.lu_info, one per node, for the
 * caller to read with the rest of the status.
 */

module;

#include "feast_bridge.h"

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.feast:contour_filter;

import std;
import wwr.runtime_api;     // wwrStream_t, wwrGetLastError, wwrSuccess
import wwr.blas;            // wwrblasHandle_t, WWRBLAS_*, wwrblasFillMode_t, wwrblasOperation_t
import wwr.wrappers.common; // real_fp, RealToComplexType
import wwr.wrappers.blas;   // getrfBatched, getrsBatched
import :buffer_size;
export import calaman.error_handling; // Status -- the cross-domain return type

export namespace calaman {

/**
 * @brief Build the resolvents Z_e I - A and LU-factor them in one batched call.
 *
 * @param uplo    Which triangle of @p d_A is stored; the other is never read.
 * @param contour From make_feast_contour.
 * @param s       Workspace, from make_feast_slices with the same n and node count.
 */
template<wwr::real_fp T>
Status feast_factor_resolvents(wwr::wwrblasHandle_t cublas_handle, wwr::wwrStream_t stream,
                               const wwr::wwrblasFillMode_t uplo, const int n, const T *d_A,
                               const int lda, const device::FeastContour<T> &contour,
                               const FeastSlices<T> &s) {
  using C = wwr::RealToComplexType<T>;
  const bool lower = (uplo == wwr::WWRBLAS_FILL_MODE_LOWER);

  device::feast_resolvents(stream, lower, n, d_A, lda, contour, s.resolvents, s.resolvent_stride);
  device::feast_pointer_array(stream, s.resolvents, s.resolvent_stride, contour.count,
                              s.resolvent_ptrs);
  device::feast_pointer_array(stream, s.rhs, s.rhs_stride, contour.count, s.rhs_ptrs);
  // A kernel-launch failure is a runtime-domain error, carried as such.
  CLM_TRY(wwr::wwrGetLastError());

  CLM_TRY(wwr::getrfBatched<C>(cublas_handle, n, s.resolvent_ptrs, n, s.ipiv, s.lu_info,
                               contour.count));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief d_out = rho(A) d_Y, using the factors feast_factor_resolvents left in @p s.
 *
 * @param d_Y   n x m0, leading dimension n. Not modified.
 * @param d_out n x m0, leading dimension n. May not alias @p d_Y's blocks in @p s.
 */
template<wwr::real_fp T>
Status feast_apply_filter(wwr::wwrblasHandle_t cublas_handle, wwr::wwrStream_t stream, const int n,
                          const int m0, const T *d_Y, const device::FeastContour<T> &contour,
                          const FeastSlices<T> &s, T *d_out) {
  using C = wwr::RealToComplexType<T>;
  const std::size_t nm = static_cast<std::size_t>(n) * static_cast<std::size_t>(m0);

  device::feast_broadcast(stream, nm, d_Y, contour.count, s.rhs, s.rhs_stride);
  CLM_TRY(wwr::wwrGetLastError());

  // getrsBatched's info is a host int and reports only invalid arguments; the
  // factorization itself is checked through s.lu_info.
  int info = 0;
  CLM_TRY(wwr::getrsBatched<C>(cublas_handle, wwr::WWRBLAS_OP_N, n, m0, s.resolvent_ptrs, n, s.ipiv,
                               s.rhs_ptrs, n, &info, contour.count));
  if (info != 0) {
    return wwr::WWRBLAS_STATUS_INTERNAL_ERROR;
  }

  device::feast_accumulate(stream, nm, contour, s.rhs, s.rhs_stride, d_out);
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
