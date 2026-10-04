/**
 * @file interface.cppm
 * @brief Primary interface for calaman.inv_sqrt -- the symmetric (Loewdin)
 *        inverse square root X = S^{-1/2} of a real symmetric PSD matrix
 *
 * Given a real-symmetric, positive-(semi)definite S (column-major, packed
 * leading dimension n), builds the symmetric root
 *
 *     X = S^{-1/2} = U diag(lambda^{-1/2}) U^T
 *
 * where S = U diag(lambda) U^T is the symmetric eigendecomposition. This is the
 * Loewdin orthogonalizer: it turns a generalized eigenproblem F C = S C eps into
 * a standard one (F' = X^T F X, C = X C'), and the same S^{-1/2} shape recurs for
 * any symmetric PSD operand (an overlap, a density-fitting Coulomb metric), which
 * is why it lives here rather than beside one caller. It is the symmetric cousin
 * of calaman.orthogonalize (that one is the QR thin-Q of a general matrix; this
 * one is the eigen-root of a symmetric one).
 *
 * The computation, all on the GPU over the WarpWraps wrappers:
 *   1. syevd(S) -> ascending eigenvalues lambda and eigenvectors U. syevd
 *      overwrites its input, so S is first copied into the caller's U scratch --
 *      @p s_dev is left untouched.
 *   2. inverse_sqrt kernel: Lambda <- diag(lambda^{-1/2}) in place, flooring
 *      modes at or below @p eigenvalue_floor to 0 (see Conditioning).
 *   3. M = U diag(Lambda^{-1/2})   (dgmm, side = RIGHT: column scaling).
 *   4. X = M U^T                   (gemm, transb = T). X is symmetric.
 *
 * Allocates NOTHING: every intermediate -- U, M, the eigenvalue vector, the
 * syevd workspace, the info flag -- is caller-owned and passed in, as in
 * calaman.orthogonalize. inv_sqrt_bufferSize sizes the syevd workspace.
 *
 * Conditioning. The kernel floors: a mode at or below @p eigenvalue_floor
 * contributes 0 instead of lambda^{-1/2}, which drops it and keeps X free of
 * NaN/Inf for a zero or negative eigenvalue. The decision is per-element, so
 * steps 2-4 are fully stream-ordered -- the routine NEVER synchronizes. syevd's
 * convergence flag is handed back to the caller in @p info_device to inspect
 * (the calaman.orthogonalize convention), rather than read back on an internal
 * host round-trip. Consequently X^T S X == I only when every eigenvalue clears
 * the floor (a well-conditioned S); for a singular or indefinite S the dropped
 * modes make X^T S X a rank-deficient projector -- expected, since such an S has
 * no true S^{-1/2}. This is not full canonical orthogonalization: X stays square.
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Needs calaman::Status visible at expansion, which the export
// import below supplies. inv_sqrt_bridge.h declares the device launcher.
#include "error_handling/error_macros.h"
#include "inv_sqrt_bridge.h"

export module calaman.inv_sqrt;

import std;
import wwr.blas;            // wwrblasHandle_t, WWRBLAS_OP_*/SIDE_*/FILL_MODE_*
import wwr.solver;          // wwrsolverDnHandle_t, WWRSOLVER_EIG_MODE_VECTOR
import wwr.runtime_api;     // wwrStream_t, wwrMemcpyAsync, wwrMemcpyDeviceToDevice, wwrSuccess
import wwr.wrappers.common; // real_fp
import wwr.wrappers.blas;   // dgmm, gemm
import wwr.wrappers.solver; // syevd, syevd_bufferSize
import calaman.common;      // kOne, kZero

// export import, not a plain import: inv_sqrt and its _bufferSize RETURN
// calaman::Status, so a consumer must see Status's member functions.
export import calaman.error_handling; // Status -- the cross-domain return type

export namespace calaman {

/**
 * @brief Query the syevd workspace inv_sqrt needs, in elements of T
 *
 * @tparam T Element type (float or double)
 * @param cusolver GPU dense-solver handle
 * @param n Matrix order
 * @param lwork Output: required workspace in elements of T (0 when n <= 0)
 * @return Success, or the solver query's Status
 */
template<wwr::real_fp T>
Status inv_sqrt_bufferSize(wwr::wwrsolverDnHandle_t cusolver, const int n, int *lwork)
{
    if (n <= 0) {
        *lwork = 0;
        return wwr::wwrSuccess;
    }
    return wwr::syevd_bufferSize<T>(cusolver, wwr::WWRSOLVER_EIG_MODE_VECTOR,
                                    wwr::WWRBLAS_FILL_MODE_LOWER, n, static_cast<T *>(nullptr), n,
                                    static_cast<T *>(nullptr), lwork);
}

/**
 * @brief Build the symmetric inverse square root X = S^{-1/2} on the GPU
 *
 * On success @p x_dev holds the n x n symmetric root and satisfies X^T S X == I
 * to floating-point tolerance for a well-conditioned S; @p s_dev is not modified.
 * Every matrix operand is a RAW column-major device pointer with packed leading
 * dimension n (element (i,j) at i + j*n). Only the lower triangle of @p s_dev is
 * read (uplo = LOWER). Bind both handles to @p stream before calling.
 *
 * Allocates nothing -- every intermediate is caller-owned scratch:
 *   - @p u_scratch / @p m_scratch: n*n elements each (eigenvectors U, product M).
 *   - @p eigenvalues_dev: n elements, overwritten with Lambda then Lambda^{-1/2}.
 *   - @p syevd_workspace / @p syevd_lwork: sized by inv_sqrt_bufferSize<T>.
 *   - @p info_device: 1 int, syevd's convergence flag, for the caller to inspect
 *     (the routine does not read it back; a non-zero value means syevd did not
 *     converge and @p x_dev is not meaningful).
 *
 * @tparam T Element type (float or double)
 * @param cublas GPU BLAS handle (dgmm, gemm); bind to @p stream first
 * @param cusolver GPU dense-solver handle (syevd); bind to @p stream first
 * @param stream Stream the copy and kernel are enqueued on
 * @param n Matrix order
 * @param s_dev Symmetric operand (n x n, column-major); read-only
 * @param x_dev Out: n x n symmetric inverse square root
 * @param u_scratch Scratch for the eigenvectors U (n x n)
 * @param m_scratch Scratch for M = U diag(Lambda^{-1/2}) (n x n)
 * @param eigenvalues_dev Device eigenvalue buffer (n), overwritten
 * @param syevd_workspace syevd's device workspace
 * @param syevd_lwork Size of @p syevd_workspace in elements of T
 * @param info_device syevd's convergence flag (1 int), handed back to the caller
 * @param eigenvalue_floor Eigenvalues at or below this are dropped; default 1e-10
 * @return Success, or the first failing Status from the copy/syevd/kernel/dgmm/gemm
 */
template<wwr::real_fp T>
Status inv_sqrt(wwr::wwrblasHandle_t cublas, wwr::wwrsolverDnHandle_t cusolver,
                wwr::wwrStream_t stream, const int n, const T *s_dev, T *x_dev, T *u_scratch,
                T *m_scratch, T *eigenvalues_dev, T *syevd_workspace, const int syevd_lwork,
                int *info_device, const T eigenvalue_floor = T{1e-10})
{
    // A 0x0 (or empty) operand has a 0x0 X; nothing to diagonalize.
    if (n <= 0) {
        return wwr::wwrSuccess;
    }
    const std::size_t matrix_bytes =
        sizeof(T) * static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

    // 1. syevd overwrites its input with the eigenvectors, so diagonalize a copy
    //    of S in u_scratch -- the caller's s_dev is untouched.
    if (const wwr::wwrError_t e = wwr::wwrMemcpyAsync(u_scratch, s_dev, matrix_bytes,
                                                      wwr::wwrMemcpyDeviceToDevice, stream);
        e != wwr::wwrSuccess) {
        return e;
    }
    CLM_TRY(wwr::syevd<T>(cusolver, wwr::WWRSOLVER_EIG_MODE_VECTOR, wwr::WWRBLAS_FILL_MODE_LOWER, n,
                          u_scratch, n, eigenvalues_dev, syevd_workspace, syevd_lwork, info_device));

    // 2. Lambda <- diag(lambda^{-1/2}), flooring near-null / non-positive modes.
    device::inverse_sqrt<T>(stream, n, eigenvalues_dev, eigenvalue_floor);
    CLM_TRY(wwr::wwrGetLastError());

    // 3. M = U diag(Lambda^{-1/2}): scale column j of U by lambda_j^{-1/2}.
    CLM_TRY(wwr::dgmm<T, int>(cublas, wwr::WWRBLAS_SIDE_RIGHT, n, n, u_scratch, n, eigenvalues_dev, 1,
                              m_scratch, n));

    // 4. X = M U^T. X is symmetric by construction.
    return wwr::gemm<T, int>(cublas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n, n, &kOne<T>,
                             m_scratch, n, u_scratch, n, &kZero<T>, x_dev, n);
}

// Instantiated once in instantiations.cpp; these keep consumers from
// re-instantiating the bodies above.
extern template Status inv_sqrt_bufferSize<float>(wwr::wwrsolverDnHandle_t, const int, int *);
extern template Status inv_sqrt_bufferSize<double>(wwr::wwrsolverDnHandle_t, const int, int *);

extern template Status inv_sqrt<float>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                       wwr::wwrStream_t, const int, const float *, float *, float *,
                                       float *, float *, float *, const int, int *, const float);
extern template Status inv_sqrt<double>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                        wwr::wwrStream_t, const int, const double *, double *,
                                        double *, double *, double *, double *, const int, int *,
                                        const double);

} // namespace calaman
