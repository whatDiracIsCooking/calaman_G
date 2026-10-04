/**
 * @file ormhr.cppm
 * @brief The calaman.ormhr module -- multiply a general matrix by the orthogonal
 *        Q from ?gehrd, LAPACK's ?ormhr
 *
 * Overwrites C (m-by-n, column-major) with one of Q*C, Q^T*C, C*Q, C*Q^T, where
 * Q = H(ilo) H(ihi-1) ... is the orthogonal matrix ?gehrd formed when it reduced
 * a matrix to upper Hessenberg form over the window [ilo, ihi]. @p A and @p tau
 * are the packed reflectors ?gehrd (or ?gehd2) left behind, unchanged here. The
 * order of Q is nq = m for Side::L, n for Side::R, and the reflectors it is built
 * from number nh = ihi - ilo. Templated over float, double.
 *
 * A pure INDEX SHIFT over ?ormqr, not a kernel: the gehrd reflectors occupy
 * columns ilo..ihi-1 of A below the subdiagonal -- reflector i has v(i+1)=1 and
 * its tail in A(i+2:ihi, i) -- which is exactly the geqrf layout ?ormqr reads,
 * shifted down one row. So ormhr hands ?ormqr the nh reflectors at A(ilo+1, ilo)
 * with tau(ilo), applied to the C submatrix that starts one past ilo on the side
 * Q multiplies. The reflectors span only the nh window rows, so ?ormqr sees that
 * extent -- C(ilo+1, 1) with mi = nh rows for Side::L, C(1, ilo+1) with ni = nh
 * columns for Side::R. ?unmqr is the complex counterpart; over float/double
 * ?ormqr suffices, as ?gehrd itself is real-only here.
 *
 * Allocation-free shipped surface (CLAUDE.md): A, tau, C and the solver workspace
 * are caller-provided device pointers; ormhr_work_len queries the length the
 * ?ormqr apply needs. Takes a wwr::wwrsolverDnHandle_t (not a BLAS handle): the
 * work is entirely ?ormqr, a solver routine. ilo/ihi are 1-based, the convention
 * ?gehrd/?gehd2 report, so a gebal -> gehrd -> ormhr pipeline forwards them.
 *
 * Usage:
 *   import calaman.ormhr;
 *   import wwr.solver;   // wwrsolverDnHandle_t, wwrsolverDnCreate/SetStream
 *   import calaman.common; // Side, Trans
 *   // A: packed reflectors from gehrd (nq-by-nq, lda); tau: length nq-1;
 *   // C: m-by-n device matrix, ldc; swork: length ormhr_work_len(...); info: int
 *   calaman::ormhr<double>(solver, Side::L, Trans::T, m, n, ilo, ihi, A, lda, tau,
 *                          C, ldc, swork, lwork, info);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the export import
// below supplies.
#include "error_handling/error_macros.h"

export module calaman.ormhr;

import wwr.blas;            // wwrblasSideMode_t, wwrblasOperation_t, WWRBLAS_SIDE/OP_*
import wwr.solver;          // wwrsolverDnHandle_t, wwrsolverStatus_t, WWRSOLVER_*
import wwr.runtime_api;     // wwrErrorInvalidValue -- the illegal-argument Status
import wwr.wrappers.solver; // ormqr, ormqr_bufferSize
import calaman.common;      // Side, Trans
import std;                 // std::max, std::min

// export import, not a plain import: ormhr RETURNS calaman::Status, so a consumer
// of `import calaman.ormhr;` must see Status's member functions, not just its
// name -- the re-export geqp3/gehd2 do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

/// @brief Length (in T elements) of the ?ormqr workspace ormhr needs
///
/// ormhr's only work is the ?ormqr apply, so this is one ormqr_bufferSize query
/// at the shifted sizes (side = @p side, the shifted mi/ni, k = nh reflectors).
/// It runs on the live solver handle and returns 1 (a harmless floor) on a
/// degenerate case (nh <= 0, empty C) or a query failure; the driver re-checks
/// the status when it sizes the real workspace. ilo/ihi are 1-based.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param solver GPU solver handle; the *_bufferSize query runs on it
/// @param side Side::L (Q on the left) or Side::R (Q on the right)
/// @param m Row count of C
/// @param n Column count of C
/// @param ilo 1-based first index of the gehrd window
/// @param ihi 1-based last index of the gehrd window
/// @param lda Leading dimension of A (>= max(1, nq), nq = m for L, n for R)
/// @param ldc Leading dimension of C (>= max(1, m))
export template<typename T>
int ormhr_work_len(wwr::wwrsolverDnHandle_t solver, const Side side, const int m, const int n,
                   const int ilo, const int ihi, const int lda, const int ldc) {
  const int nh = ihi - ilo;
  if (m <= 0 || n <= 0 || nh <= 0) {
    return 1;
  }
  const bool left = side == Side::L;
  const int mi = left ? nh : m;
  const int ni = left ? n : nh;
  const wwr::wwrblasSideMode_t s = left ? wwr::WWRBLAS_SIDE_LEFT : wwr::WWRBLAS_SIDE_RIGHT;
  int lwork = 0;
  if (wwr::ormqr_bufferSize<T>(solver, s, wwr::WWRBLAS_OP_N, mi, ni, nh, nullptr, lda, nullptr,
                               nullptr, ldc, &lwork) != wwr::WWRSOLVER_STATUS_SUCCESS) {
    return 1;
  }
  return std::max(lwork, 1);
}

/// @brief Multiply C by the orthogonal Q from ?gehrd (LAPACK ?ormhr)
///
/// Overwrites C with Q*C, Q^T*C, C*Q or C*Q^T per @p side and @p trans, Q being
/// the product of the nh = ihi-ilo reflectors ?gehrd packed into @p A below the
/// subdiagonal of columns ilo..ihi-1, with scalars @p tau. Forwards the shifted
/// reflectors A(ilo+1, ilo), tau(ilo) and the shifted C submatrix to one
/// wwr::ormqr; a failing ormqr surfaces as its own solver-domain Status.
///
/// Validates as LAPACK's ?ormhr INFO contract: an illegal side/trans/m/n/ilo/ihi
/// /lda/ldc returns wwrErrorInvalidValue and touches nothing. Quick return (a
/// success writing nothing) when m == 0, n == 0, or nh == 0. nq = m for Side::L,
/// n for Side::R is the order of Q. Trans::C is rejected -- this is the real path.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param solver GPU solver handle; A, tau, C, work live on its stream
/// @param side Side::L for Q*C / Q^T*C, Side::R for C*Q / C*Q^T
/// @param trans Trans::N applies Q, Trans::T applies Q^T
/// @param m Row count of C (>= 0)
/// @param n Column count of C (>= 0)
/// @param ilo 1-based first index of the gehrd window; 1 <= ilo <= max(1, nq)
/// @param ihi 1-based last index of it; min(ilo, nq) <= ihi <= nq
/// @param A Device matrix, nq by nq, column-major; the packed gehrd reflectors (read-only)
/// @param lda Leading dimension of A (>= max(1, nq))
/// @param tau Device array, length >= nq-1; the reflector scalars (read-only)
/// @param C Device matrix, m by n, column-major; overwritten with the product
/// @param ldc Leading dimension of C (>= max(1, m))
/// @param work Device workspace, length >= ormhr_work_len(...)
/// @param lwork Length of @p work in T elements
/// @param info Device int; the ormqr devInfo (0 on success)
/// @return The ormqr Status, otherwise WWRBLAS_STATUS_SUCCESS
export template<typename T>
Status ormhr(wwr::wwrsolverDnHandle_t solver, const Side side, const Trans trans, const int m,
             const int n, const int ilo, const int ihi, const T *A, const int lda, const T *tau,
             T *C, const int ldc, T *work, const int lwork, int *info) {
  const bool left = side == Side::L;
  const int nq = left ? m : n;

  // Argument validation, in LAPACK's ?ormhr INFO order. Trans::C is not a real
  // operation; the real path takes only N / T.
  if ((side != Side::L && side != Side::R) || (trans != Trans::N && trans != Trans::T) || m < 0 ||
      n < 0 || ilo < 1 || ilo > std::max(1, nq) || ihi < std::min(ilo, nq) || ihi > nq ||
      lda < std::max(1, nq) || ldc < std::max(1, m)) {
    return wwr::wwrErrorInvalidValue;
  }

  // Quick return: empty C, or a window with no reflectors (nh = ihi - ilo).
  const int nh = ihi - ilo;
  if (m == 0 || n == 0 || nh == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  // Shift onto the ?ormqr the gehrd reflectors describe: nh reflectors at
  // A(ilo+1, ilo) with tau(ilo), applied to the C submatrix one past ilo on Q's
  // side. The reflectors span only the nh = ihi-ilo window rows, so ?ormqr sees
  // exactly that extent -- Side::L: mi = nh rows at C(ilo+1, 1), ni = n; Side::R:
  // mi = m, ni = nh cols at C(1, ilo+1). Taking mi = m-ilo would make ?ormqr read
  // the rows below ihi of the reflector columns as reflector data, which they are
  // not. Offsets are 0-based element counts into the column-major buffers:
  // A(ilo+1, ilo) = A + (ilo-1)*lda + ilo, tau(ilo) = tau + (ilo-1).
  const int mi = left ? nh : m;
  const int ni = left ? n : nh;
  const std::size_t a_off = static_cast<std::size_t>(ilo - 1) * lda + static_cast<std::size_t>(ilo);
  const std::size_t t_off = static_cast<std::size_t>(ilo - 1);
  const std::size_t c_off =
      left ? static_cast<std::size_t>(ilo) : static_cast<std::size_t>(ilo) * ldc;

  const wwr::wwrblasSideMode_t s = left ? wwr::WWRBLAS_SIDE_LEFT : wwr::WWRBLAS_SIDE_RIGHT;
  const wwr::wwrblasOperation_t op = trans == Trans::N ? wwr::WWRBLAS_OP_N : wwr::WWRBLAS_OP_T;

  // ormqr takes a non-const A/tau; it does not modify them (the geqrf layout is
  // read-only input here), so the const_cast is safe and keeps ormhr's surface
  // read-only in A and tau, as LAPACK's ?ormhr promises.
  CLM_TRY(wwr::ormqr<T>(solver, s, op, mi, ni, nh, const_cast<T *>(A) + a_off, lda,
                        const_cast<T *>(tau) + t_off, C + c_off, ldc, work, lwork, info));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
