/**
 * @file gehd2.cppm
 * @brief The calaman.gehd2 module -- unblocked reduction of a general matrix to
 *        upper Hessenberg form, LAPACK's ?gehd2
 *
 * Reduces a real n-by-n column-major matrix A to upper Hessenberg form H by an
 * orthogonal similarity Q^T * A * Q = H, overwriting A the standard LAPACK way:
 * H in the upper triangle and first subdiagonal, and reflector i's tail below
 * the subdiagonal of column i, with the Householder scalars in @p tau. Q is the
 * product H(ilo) H(ilo+1) ... H(ihi-1); each H(i) = I - tau*v*v^T has
 * v(1:i)=0, v(i+1)=1, v(i+2:ihi) stored in A(i+2:ihi,i) and v(ihi+1:n)=0. A is
 * assumed already upper triangular outside the window [ilo, ihi] (normally set
 * by a prior calaman.gebal; else 1 and n). Templated over `float` and `double`.
 *
 * It is a HOST COMPOSITION, not a kernel, and the slim sibling of calaman.laqp2:
 * each step reaches for larfg (the reflector on A(i+1:ihi,i)) then larf1f twice
 * -- a right apply to A(1:ihi,i+1:ihi) and a left apply to A(i+1:ihi,i+1:n).
 * larf1f, not larf, because larfg emits the implicit-unit-head reflector larf1f
 * expects, so there is no set-to-one/restore dance around the stored beta.
 *
 * Allocation-free shipped surface (CLAUDE.md): tau is a caller-provided device
 * array (length >= n-1; only tau[ilo-1:ihi-1] is written, as LAPACK's ?gehd2 --
 * unlike ?gehrd -- does not zero the rest) and work a device buffer of length
 * >= n. ilo/ihi are 1-based, the convention calaman.gebal reports and ?gehd2
 * takes, so the gebal -> gehd2 pipeline forwards them unchanged.
 *
 * Requires the handle's DEFAULT (host) pointer mode, like larfg/larf1f. The one
 * host<->device sync per step is larfg's (reading alpha); the tau scalars ride a
 * single host buffer flushed to the device once at the end. Complex ?gehd2 is a
 * deliberate later extension, for the reason larfg documents (a complex tau and
 * the conjugated reflector differ materially from a trivial instantiation).
 *
 * Usage:
 *   import calaman.gehd2;
 *   import wwr.blas;   // wwrblasHandle_t, wwrblasCreate
 *   // d_A: n x n device matrix, lda; d_tau: length n-1; d_work: length n
 *   calaman::gehd2<double>(handle, n, 1, n, d_A, lda, d_tau, d_work);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.gehd2;

import wwr.blas;          // wwrblasHandle_t, wwrblasGetStream, WWRBLAS_STATUS_SUCCESS
import wwr.runtime_api;   // wwrStream_t, wwrMemcpyAsync, wwrStreamSynchronize, wwrErrorInvalidValue
import calaman.larfg;     // calaman::larfg
import calaman.larf1f;    // calaman::larf1f, calaman::Side
import std;               // std::min, std::max, std::size_t, std::vector

// export import, not a plain import: gehd2 RETURNS calaman::Status, so a consumer
// of `import calaman.gehd2;` must see Status's member functions, not just its
// name -- the same re-export laqp2/larf1f do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

/// @brief Reduce A to upper Hessenberg form by an orthogonal similarity (?gehd2)
///
/// Computes Q^T*A*Q = H in place: H in A's upper triangle and first subdiagonal,
/// reflector i's tail below the subdiagonal of column i, tau[i-1] its scalar.
/// A is assumed upper triangular in rows/columns 1:ilo-1 and ihi+1:n. Runs as a
/// host loop over larfg + two larf1f applies; a failing step short-circuits and
/// returns its Status. Validates arguments as LAPACK's INFO contract does,
/// returning wwrErrorInvalidValue on the first illegal one; a window with no
/// reflectors (ihi <= ilo) is a success that writes nothing.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle GPU BLAS handle in host pointer mode; A, tau, work live on its device
/// @param n Order of the matrix A (>= 0)
/// @param ilo 1-based first index of the active window; 1 <= ilo <= max(1,n)
/// @param ihi 1-based last index of it; min(ilo,n) <= ihi <= n
/// @param A Device matrix, n by n, column-major, overwritten with H and the reflectors
/// @param lda Leading dimension of A (>= max(1,n))
/// @param tau Device array, length >= n-1; tau[ilo-1:ihi-1] receive the reflector scalars
/// @param work Device workspace, length >= n; the larf1f intermediate per step
/// @return The Status of the failing step, otherwise WWRBLAS_STATUS_SUCCESS
export template<typename T>
Status gehd2(wwr::wwrblasHandle_t handle, const int n, const int ilo, const int ihi, T *A,
             const int lda, T *tau, T *work) {
  // Argument validation, in LAPACK's INFO order (-1, -2, -3, -5).
  if (n < 0 || ilo < 1 || ilo > std::max(1, n) || ihi < std::min(ilo, n) || ihi > n ||
      lda < std::max(1, n)) {
    return wwr::wwrErrorInvalidValue;
  }

  // Reflectors H(ilo) .. H(ihi-1): nothing to do for an empty or singleton window.
  const int num_reflectors = ihi - ilo;
  if (num_reflectors < 1) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  // Order every scalar read/write on the handle's own stream, so the tau flush
  // follows the caller's uploads and this routine's BLAS work -- the larfg
  // discipline.
  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(handle, &stream));

  // tau rides a persistent host buffer and is flushed to the device once at the
  // end: a per-step async copy would read a stack scalar that has gone out of
  // scope before the copy runs (the bug larfg's beta-writeback warns about).
  std::vector<T> tau_host(static_cast<std::size_t>(num_reflectors), T{});

  const int ihi0 = ihi - 1; // 0-based last active index

  for (int i0 = ilo - 1; i0 < ihi0; ++i0) {
    const int order = ihi0 - i0; // reflector order = ihi - i (1-based i = i0 + 1)

    // v = A(i+1, i): the reflector lives in column i0, from row i0+1 down. larfg
    // overwrites A(i0+1,i0) with beta and the tail A(i0+2:ihi,i0) with v(2:); it
    // reads the tail from v + 1 (contiguous), the same form laqp2 passes.
    T *v = A + static_cast<std::size_t>(i0) * lda + (i0 + 1);
    T host_tau{};
    T host_beta{};
    CLM_TRY(larfg<T>(handle, order, v, v + 1, 1, &host_tau, &host_beta));
    tau_host[static_cast<std::size_t>(i0 - (ilo - 1))] = host_tau;

    // Apply H(i) to A(1:ihi, i+1:ihi) from the right. larf1f treats v(1) as the
    // implicit 1 and never reads the stored beta, so no set-to-one dance.
    CLM_TRY(larf1f<T>(handle, Side::R, ihi, order, v, 1, host_tau,
                      A + static_cast<std::size_t>(i0 + 1) * lda, lda, work));

    // Apply H(i) to A(i+1:ihi, i+1:n) from the left.
    CLM_TRY(larf1f<T>(handle, Side::L, order, n - i0 - 1, v, 1, host_tau,
                      A + static_cast<std::size_t>(i0 + 1) * lda + (i0 + 1), lda, work));
  }

  // Flush the reflector scalars into tau[ilo-1 .. ihi-2] on the stream, after the
  // BLAS work, and block until the copy lands (tau_host is a local).
  CLM_TRY(wwr::wwrMemcpyAsync(tau + (ilo - 1), tau_host.data(), tau_host.size() * sizeof(T),
                              wwr::wwrMemcpyHostToDevice, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
