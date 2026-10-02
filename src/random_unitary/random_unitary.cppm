/**
 * @file random_unitary.cppm
 * @brief Primary interface for calaman.random_unitary -- a random orthogonal /
 *        unitary matrix via QR of a Gaussian matrix
 *
 * One host routine that fills an n-by-n matrix with independent standard-normal
 * entries and then overwrites it with its own Q factor, so the result is a
 * random matrix with orthonormal columns -- orthogonal for a real T, unitary for
 * a complex one. It is the composition of two shipped pieces: the device-side
 * Gaussian fill from wwr.extension.random_normal, then calaman.orthogonalize
 * (geqrf, then orgqr/ungqr) to expand the explicit Q. A `bufferSize` companion
 * sizes the one device scratch buffer the QR needs.
 *
 * No device code and no backend branch live here: the fill is one call into
 * WarpWraps's random_normal extension (which owns the per-backend .cu), and the
 * QR is calaman.orthogonalize. The real/complex split is entirely inside those
 * two -- random_normal draws two independent normals per complex element, and
 * orthogonalize's own `if constexpr` picks orgqr vs ungqr -- so this file is one
 * code path for all four usual_fp types.
 *
 * Distribution: Q is the Q of a Ginibre (iid-Gaussian) matrix, so it is
 * Haar-distributed up to the per-column sign (real) or phase (complex) that the
 * QR convention fixes -- the standard "take QR of a Gaussian" construction, with
 * no extra phase correction applied. That is enough for an unbiased orthonormal
 * basis / a generic test unitary; it is not a bitwise-canonical Haar sample.
 *
 * The states array is the CALLER's: it is an array of n*n generator states that
 * must already have been initialized (by wwr.extension.init_state's init_state)
 * before this call, and random_normal advances them, so feeding the same array
 * to a later call continues the streams rather than repeating them. The Gaussian
 * fill runs on the stream currently set on the solver handle, so the fill and
 * the QR that reads its output are ordered on one stream.
 *
 * The workspace is the SAME single buffer calaman.orthogonalize uses, sized by
 * random_unitary_bufferSize (which forwards to orthogonalize_bufferSize for an
 * n-by-n matrix); see calaman.orthogonalize for its tau-then-scratch layout.
 *
 * Templated over float, double, wwrFloatComplex and wwrDoubleComplex -- the four
 * usual_fp types, instantiated once in instantiations.cpp behind the interface's
 * `extern template` list so a consumer never re-instantiates the bodies.
 *
 * Usage:
 *   import calaman.random_unitary;
 *   import wwr.solver;                 // wwrsolverDnHandle_t, wwrsolverCreate
 *   import wwr.rand;                   // wwrrandState
 *   import wwr.extension.init_state;   // init_state (seed the states first)
 *   wwr::wwrsolverDnHandle_t handle{};
 *   wwr::wwrsolverDnCreate(&handle);
 *   wwr::wwrsolverDnSetStream(handle, stream);
 *   int lwork = 0;
 *   calaman::random_unitary_bufferSize<double>(handle, n, &lwork);
 *   // allocate d_Q (n*n, column-major), d_work (lwork doubles), two int infos,
 *   // and states (n*n); then init_state(stream, n*n, states, seed);
 *   calaman::random_unitary<double>(handle, n, d_Q, d_work, states, lwork,
 *                                   d_info_geqrf, d_info_gqr);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.random_unitary;

import std;
import wwr.solver;                   // wwrsolverDnHandle_t, wwrsolverStatus_t, wwrsolverDnGetStream
import wwr.runtime_api;              // wwrStream_t, wwrGetLastError, wwrSuccess
import wwr.rand;                     // wwrrandState
import wwr.wrappers.common;          // usual_fp + wwrFloatComplex/wwrDoubleComplex
import wwr.extension.random_normal;  // random_normal (the Gaussian fill)
import calaman.orthogonalize;        // orthogonalize (+ _bufferSize)

// export import, not a plain import: random_unitary and its _bufferSize RETURN
// calaman::Status, so a consumer must see Status's member functions, not just
// its name -- the same re-export diff_norm does. (calaman.orthogonalize already
// re-exports it, but name this dependency directly, not through that edge.)
export import calaman.error_handling; // Status -- the cross-domain return type

export namespace calaman {

/**
 * @brief Query the workspace random_unitary needs, in elements of T
 *
 * Forwards to orthogonalize_bufferSize for the n-by-n matrix: the random fill
 * needs no scratch of its own, so the buffer is exactly the QR's (tau followed
 * by the geqrf / orgqr-ungqr scratch). See calaman.orthogonalize for the layout.
 *
 * @tparam T Element type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle GPU dense-solver handle
 * @param n Matrix dimension (n-by-n)
 * @param lwork Output: required workspace in elements of T
 * @return Success, or the first failing Status
 */
template <wwr::usual_fp T>
Status random_unitary_bufferSize(
    wwr::wwrsolverDnHandle_t handle,
    const int n,
    int* lwork)
{
    return orthogonalize_bufferSize<T>(handle, n, n, lwork);
}

/**
 * @brief Generate a random orthogonal / unitary matrix via QR of a Gaussian fill
 *
 * Overwrites @p d_Q with an n-by-n matrix whose columns are orthonormal:
 * 1. Fill @p d_Q with n*n independent standard-normal entries via random_normal.
 * 2. Orthogonalize @p d_Q in place with calaman.orthogonalize (geqrf, then
 *    orgqr for a real T or ungqr for a complex one).
 *
 * The fill runs on the stream currently set on @p handle, so step 2 sees step 1's
 * output. The two info outputs carry geqrf's and orgqr/ungqr's per-call
 * diagnostics; see calaman.orthogonalize for its early-return-on-failure
 * behaviour.
 *
 * @tparam T Element type (float, double, wwrFloatComplex, wwrDoubleComplex)
 * @param handle GPU dense-solver handle (its stream carries the Gaussian fill)
 * @param n Matrix dimension (n-by-n)
 * @param d_Q Output: n-by-n matrix (column-major, ld=n) with orthonormal columns
 * @param d_work Device workspace (lwork elements of T; includes tau storage)
 * @param states Device array of n*n generator states, already initialized by
 *               wwr.extension.init_state's init_state; advanced by this call
 * @param lwork Workspace size from random_unitary_bufferSize
 * @param d_info_geqrf Device buffer for geqrf's info output (single int)
 * @param d_info_gqr Device buffer for orgqr/ungqr's info output (single int)
 * @return Success, or the first failing Status -- the stream query, the fill's
 *         launch error, or whichever QR step failed
 */
template <wwr::usual_fp T>
Status random_unitary(
    wwr::wwrsolverDnHandle_t handle,
    const int n,
    T* d_Q,
    T* d_work,
    wwr::wwrrandState* states,
    const int lwork,
    int* d_info_geqrf,
    int* d_info_gqr)
{
    // The fill must land on the handle's own stream so the QR that follows reads
    // it in order; random_normal takes a stream, orthogonalize reads the handle.
    wwr::wwrStream_t stream{};
    CLM_TRY(wwr::wwrsolverDnGetStream(handle, &stream));

    // Step 1: fill with standard normal (default scale 1 -- Q is scale-invariant,
    // so the component scaling that distinguishes unit magnitude is immaterial).
    const std::size_t count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    wwr::extension::random_normal<T>(stream, count, states, d_Q);
    // random_normal's launcher returns void, so the only way to catch a bad
    // launch is the runtime's sticky error -- checked here, as gebal does.
    CLM_TRY(wwr::wwrGetLastError());

    // Step 2: overwrite with the explicit Q of that Gaussian matrix.
    return orthogonalize<T>(handle, n, n, d_Q, d_work, lwork, d_info_geqrf, d_info_gqr);
}

// Instantiated once in instantiations.cpp; these keep consumers from
// re-instantiating the bodies above.

// Function: random_unitary_bufferSize
extern template Status random_unitary_bufferSize<float>(wwr::wwrsolverDnHandle_t, const int, int*);
extern template Status random_unitary_bufferSize<double>(wwr::wwrsolverDnHandle_t, const int, int*);
extern template Status random_unitary_bufferSize<wwr::wwrFloatComplex>(wwr::wwrsolverDnHandle_t, const int, int*);
extern template Status random_unitary_bufferSize<wwr::wwrDoubleComplex>(wwr::wwrsolverDnHandle_t, const int, int*);

// Function: random_unitary
extern template Status random_unitary<float>(wwr::wwrsolverDnHandle_t, const int, float*, float*, wwr::wwrrandState*, const int, int*, int*);
extern template Status random_unitary<double>(wwr::wwrsolverDnHandle_t, const int, double*, double*, wwr::wwrrandState*, const int, int*, int*);
extern template Status random_unitary<wwr::wwrFloatComplex>(wwr::wwrsolverDnHandle_t, const int, wwr::wwrFloatComplex*, wwr::wwrFloatComplex*, wwr::wwrrandState*, const int, int*, int*);
extern template Status random_unitary<wwr::wwrDoubleComplex>(wwr::wwrsolverDnHandle_t, const int, wwr::wwrDoubleComplex*, wwr::wwrDoubleComplex*, wwr::wwrrandState*, const int, int*, int*);

} // namespace calaman
