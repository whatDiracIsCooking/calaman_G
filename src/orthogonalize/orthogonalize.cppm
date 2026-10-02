/**
 * @file orthogonalize.cppm
 * @brief Primary interface for calaman.orthogonalize -- an orthonormal basis for
 *        the columns of a matrix, via QR
 *
 * One host routine that overwrites an m-by-n input (m >= n, column-major) with
 * its thin Q factor, so the result's columns form an orthonormal set. It is the
 * composition of two legacy cuSOLVER/hipSOLVER calls: geqrf factorises in place
 * (the lower triangle becomes the Householder vectors, the upper the R that is
 * then discarded), and orgqr (real) or ungqr (complex) expands those reflectors
 * back into the explicit thin Q. A `bufferSize` companion sizes the single
 * device scratch buffer the pair needs.
 *
 * No device code lives here. geqrf/orgqr/ungqr are stock dense-solver calls, so
 * this reaches the OUTERMOST WarpWraps layer that does the job -- the type-safe
 * dispatch wrappers in `wwr.wrappers.solver` (see CLAUDE.md, "prefer the
 * outermost layer"). There is no .cu, no launcher bridge: unlike calaman.lacpy,
 * nothing backend-specific is written. The real/complex split is a compile-time
 * `if constexpr` on the element type, exactly as the wrappers themselves
 * dispatch s/d vs c/z.
 *
 * The workspace is ONE buffer the caller allocates from
 * orthogonalize_bufferSize, laid out as:
 *   d_work[0 .. tau_size-1]        : tau, the Householder scalars
 *   d_work[tau_size .. lwork-1]    : the cuSOLVER/hipSOLVER scratch
 * where tau_size = align_up(n, 256) rounds the n scalars up to a 256-element
 * boundary so the solver scratch that follows stays aligned. The scratch span is
 * the max of what geqrf and orgqr/ungqr each ask for, since they run in
 * sequence and reuse it.
 *
 * Templated over float, double, wwrFloatComplex and wwrDoubleComplex -- the four
 * usual_fp types, instantiated once in instantiations.cpp behind the interface's
 * `extern template` list so a consumer never re-instantiates the bodies.
 *
 * Usage:
 *   import calaman.orthogonalize;
 *   import wwr.solver;          // wwrsolverDnHandle_t, wwrsolverCreate
 *   wwr::wwrsolverDnHandle_t handle{};
 *   wwr::wwrsolverCreate(&handle);
 *   int lwork = 0;
 *   calaman::orthogonalize_bufferSize<double>(handle, m, n, &lwork);
 *   // allocate d_work (lwork doubles), d_Q (m*n, column-major), two int infos
 *   calaman::orthogonalize<double>(handle, m, n, d_Q, d_work, lwork,
 *                                  d_info_geqrf, d_info_gqr);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.orthogonalize;

import std;
import wwr.solver;            // wwrsolverDnHandle_t, wwrsolverStatus_t, WWRSOLVER_STATUS_*
import wwr.wrappers.solver;   // geqrf/orgqr/ungqr (+ bufferSize); re-exports usual_fp/real_fp + complex types
import calaman.common;        // align_up (:align_up)

// export import, not a plain import: orthogonalize and its _bufferSize RETURN
// calaman::Status, so a consumer must see Status's member functions, not just
// its name -- the same re-export diff_norm does.
export import calaman.error_handling; // Status -- the cross-domain return type

export namespace calaman {

/**
 * @brief Query the workspace orthogonalize needs, in elements of T
 *
 * The buffer holds tau followed by the solver scratch:
 *   [0 .. tau_size-1]     tau (Householder scalars), tau_size = align_up(n, 256)
 *   [tau_size .. lwork-1] solver scratch = max(geqrf need, orgqr/ungqr need)
 *
 * @tparam T Element type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle GPU dense-solver handle
 * @param m Number of rows (m >= n)
 * @param n Number of columns
 * @param lwork Output: required workspace in elements of T
 * @return Success, or the first failing Status
 */
template <wwr::usual_fp T>
Status orthogonalize_bufferSize(
    wwr::wwrsolverDnHandle_t handle,
    const int m,
    const int n,
    int* lwork)
{
    int lwork_geqrf = 0, lwork_gqr = 0;
    T* dummy_ptr = static_cast<T*>(nullptr);

    CLM_TRY(wwr::geqrf_bufferSize<T>(handle, m, n, dummy_ptr, m, &lwork_geqrf));

    if constexpr (wwr::real_fp<T>) {
        CLM_TRY(wwr::orgqr_bufferSize<T>(handle, m, n, n,
                                         dummy_ptr, m,
                                         dummy_ptr, &lwork_gqr));
    } else {
        CLM_TRY(wwr::ungqr_bufferSize<T>(handle, m, n, n,
                                         dummy_ptr, m,
                                         dummy_ptr, &lwork_gqr));
    }

    const int tau_size = align_up(n, 256);
    const int solver_size = std::max(lwork_geqrf, lwork_gqr);
    *lwork = tau_size + solver_size;
    return wwr::WWRSOLVER_STATUS_SUCCESS;
}

/**
 * @brief Orthogonalize a matrix in place via QR decomposition
 *
 * Overwrites the columns of @p d_Q with an orthonormal basis for themselves:
 * 1. QR-factorize @p d_Q in place with geqrf.
 * 2. Expand the thin Q explicitly with orgqr (real) or ungqr (complex).
 *
 * For a rectangular m-by-n input (m >= n), @p d_Q is overwritten with the m-by-n
 * thin Q factor whose columns form an orthonormal set. Returns on the earliest
 * failing step; the two info outputs carry the per-call diagnostics.
 *
 * @tparam T Element type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle GPU dense-solver handle
 * @param m Number of rows (m >= n)
 * @param n Number of columns
 * @param d_Q Input/output: m-by-n matrix (column-major, ld=m); overwritten with thin Q
 * @param d_work Device workspace (lwork elements of T; includes tau storage)
 * @param lwork Workspace size from orthogonalize_bufferSize
 * @param d_info_geqrf Device buffer for geqrf's info output (single int)
 * @param d_info_gqr Device buffer for orgqr/ungqr's info output (single int)
 * @return Success, or the first failing Status (the gqr step is not enqueued if
 *         geqrf failed)
 */
template <wwr::usual_fp T>
Status orthogonalize(
    wwr::wwrsolverDnHandle_t handle,
    const int m,
    const int n,
    T* d_Q,
    T* d_work,
    const int lwork,
    int* d_info_geqrf,
    int* d_info_gqr)
{
    // d_work layout: [tau: tau_size elems (256-aligned)][solver scratch]
    const int tau_size = align_up(n, 256);
    T* tau         = d_work;
    T* solver_ws   = d_work + tau_size;

    // Step 1: QR factorization (in-place; lower triangle becomes Householder vectors)
    CLM_TRY(wwr::geqrf<T>(handle, m, n, d_Q, m, tau, solver_ws, lwork, d_info_geqrf));

    // Step 2: Expand the explicit thin Q from the Householder reflectors
    if constexpr (wwr::real_fp<T>) {
        return wwr::orgqr<T>(handle, m, n, n, d_Q, m, tau, solver_ws, lwork, d_info_gqr);
    } else {
        return wwr::ungqr<T>(handle, m, n, n, d_Q, m, tau, solver_ws, lwork, d_info_gqr);
    }
}

// Instantiated once in instantiations.cpp; these keep consumers from
// re-instantiating the bodies above.

// Function: orthogonalize_bufferSize
extern template Status orthogonalize_bufferSize<float>(wwr::wwrsolverDnHandle_t, const int, const int, int*);
extern template Status orthogonalize_bufferSize<double>(wwr::wwrsolverDnHandle_t, const int, const int, int*);
extern template Status orthogonalize_bufferSize<wwr::wwrFloatComplex>(wwr::wwrsolverDnHandle_t, const int, const int, int*);
extern template Status orthogonalize_bufferSize<wwr::wwrDoubleComplex>(wwr::wwrsolverDnHandle_t, const int, const int, int*);

// Function: orthogonalize
extern template Status orthogonalize<float>(wwr::wwrsolverDnHandle_t, const int, const int, float*, float*, const int, int*, int*);
extern template Status orthogonalize<double>(wwr::wwrsolverDnHandle_t, const int, const int, double*, double*, const int, int*, int*);
extern template Status orthogonalize<wwr::wwrFloatComplex>(wwr::wwrsolverDnHandle_t, const int, const int, wwr::wwrFloatComplex*, wwr::wwrFloatComplex*, const int, int*, int*);
extern template Status orthogonalize<wwr::wwrDoubleComplex>(wwr::wwrsolverDnHandle_t, const int, const int, wwr::wwrDoubleComplex*, wwr::wwrDoubleComplex*, const int, int*, int*);

} // namespace calaman
