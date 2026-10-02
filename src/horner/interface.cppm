/**
 * @file interface.cppm
 * @brief Primary interface for calaman.horner -- matrix polynomial evaluation
 *        by Horner's method
 *
 * Evaluates
 *
 *     p(A) = c_0 I + c_1 A + c_2 A^2 + ... + c_d A^d
 *
 * for a square column-major matrix A, using the Horner recurrence
 *
 *     P <- c_d I,   then   P <- P A + c_k I   for k = d-1, d-2, ..., 0.
 *
 * That is d matrix products for a degree-d polynomial -- the minimum for a
 * general dense evaluation, and half of what the naive "form every power, then
 * accumulate" costs (d products plus d axpy passes over n^2 elements, with the
 * powers all live at once). Horner also needs only one n-by-n scratch block
 * regardless of degree, because each step consumes the previous accumulator.
 *
 * The coefficients live on the DEVICE, in ascending order: d_coeffs[k] is the
 * coefficient of A^k. Nothing in the evaluation reads them on the host -- the
 * two diagonal updates (horner.cu, reached through horner_bridge.h) take the
 * coefficient as a device pointer -- so the whole call stays on the handle's
 * stream and coefficients computed by a previous kernel can be fed straight in.
 * This is independent of the handle's pointer mode: the gemm scalars are the
 * host constants kOne/kZero from calaman.common.
 *
 * Structurally the twin of calaman.lacpy: a host wrapper over wrapped BLAS plus
 * a device kernel, so it owns a launcher bridge (horner_bridge.h), a
 * device-compiled unit (horner.cu) and an explicit-instantiation unit
 * (instantiations.cpp). The gemm is reached at the OUTERMOST WarpWraps layer
 * that does the job -- the type-safe dispatch wrapper in wwr.wrappers.blas
 * (CLAUDE.md, "prefer the outermost layer").
 *
 * Templated over float and double. Complex ?horner is a deliberate later
 * extension, for the reasons calaman.common's constants.h documents: there is
 * no portable constexpr spelling of the kOne/kZero gemm scalars in complex.
 *
 * @note Horner is *not* the cheapest scheme at high degree. Paterson-Stockmeyer
 *       evaluates a degree-d polynomial in O(sqrt(d)) products by grouping the
 *       terms, at the cost of holding sqrt(d) powers of A; it wins for roughly
 *       d > 8. Horner is the right choice for the low degrees that show up in
 *       Pade numerators and rational filters, and whenever extra n-by-n
 *       workspace is the binding constraint.
 *
 * Usage:
 *   import calaman.horner;
 *   import wwr.blas;          // wwrblasHandle_t, wwrblasCreate
 *   wwr::wwrblasHandle_t handle{};
 *   wwr::wwrblasCreate(&handle);          // default (host) pointer mode
 *   // d_coeffs: degree+1 device scalars; d_A, d_P: n-by-n device matrices
 *   const std::size_t bytes = calaman::horner_bufferSize<double>(n);
 *   // d_work: bytes of 256-aligned device scratch
 *   calaman::horner<double>(handle, n, d_coeffs, degree, d_A, lda, d_P, ldp,
 *                           d_work, bytes);
 */

module;

#include "horner_bridge.h"
#include "error_handling/error_macros.h" // CLM_TRY / CLM_REQUIRE -- macros arrive by #include, not import

export module calaman.horner;

import std;
import wwr.blas;          // wwrblasHandle_t, wwrblasStatus_t, WWRBLAS_OP_N, WWRBLAS_STATUS_*
import wwr.runtime_api;   // wwrStream_t (the type wwrblasGetStream writes)
import wwr.wrappers.blas; // gemm
import calaman.common;    // kZero<T> / kOne<T> (:constants), WorkspaceBuilder, all_nonnull (:validation)

// export import, not a plain import: horner() RETURNS calaman::Status, whose
// member functions (ok/name/message) a consumer of this module must see, not
// just the type name -- the same reason calaman.diff_norm re-exports it. It also
// supplies the ::calaman::Status that CLM_TRY/CLM_REQUIRE name at expansion.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so each template carries its own
// `export` and the declarations below sit in the plain namespace.

// ========================================================================
// Workspace
// ========================================================================

/// @brief Device workspace required by horner(), in bytes
///
/// One n-by-n block of T, 256-byte aligned. The size does not depend on the
/// degree -- that is the point of the recurrence: each step consumes the
/// previous accumulator, so one scratch block serves any degree.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param n Order of the matrix
/// @return Bytes to allocate for horner()'s @p d_work
///
/// @note A degree-0 polynomial is just c_0 I and needs no workspace at all;
///       horner() accepts a null buffer in that case.
export template<typename T>
std::size_t horner_bufferSize(const int n) {
  const std::size_t order = static_cast<std::size_t>(n < 1 ? 1 : n);

  WorkspaceBuilder builder;
  builder.add_scratch<T>(order * order); // the ping-pong accumulator
  return builder.total();
}

// ========================================================================
// Evaluation
// ========================================================================

/// @brief Matrix polynomial by Horner's method: P = sum_{k=0}^{degree} c_k A^k
///
/// The coefficients are in ASCENDING order: d_coeffs[k] multiplies A^k, so the
/// array holds degree+1 entries and d_coeffs[degree] is the leading one. A^0 is
/// the identity, so d_coeffs[0] contributes c_0 I. P is overwritten, not
/// accumulated into. Only the n-by-n block of P is written; the padding rows
/// between n and @p ldp are left untouched, so P may be a view into a larger
/// allocation.
///
/// A gemm cannot write one of its own inputs, so the accumulator ping-pongs
/// between the caller's P and the scratch block. Each of the @p degree steps
/// flips buffers, so starting on the matching parity lands the last step in
/// @p d_P and saves a final n-by-n copy -- an odd-degree evaluation starts in
/// the workspace.
///
/// The evaluation runs entirely on the handle's stream and never synchronizes:
/// the coefficients are read on the device by the diagonal-update launchers,
/// and the gemm scalars are the host constants kOne/kZero.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle     GPU BLAS handle in default (host) pointer mode; its stream
///                   carries every operation and A, P, work live on its device
/// @param n          Order of A and P
/// @param d_coeffs   Device array of degree+1 coefficients, ascending in k
/// @param degree     Polynomial degree, at least 0
/// @param d_A        Input n-by-n column-major matrix
/// @param lda        Leading dimension of @p d_A; lda >= n
/// @param d_P        Output n-by-n column-major matrix, overwritten with p(A)
/// @param ldp        Leading dimension of @p d_P; ldp >= n
/// @param d_work     Device workspace, 256-byte aligned; may be null if degree is 0
/// @param work_bytes Size of @p d_work; at least horner_bufferSize<T>(n)
/// @return A calaman::Status: success, the first failing gemm's status, or
///         WWRBLAS_STATUS_NOT_INITIALIZED (the neutral bad-argument code
///         calaman.diff_norm documents) for bad dimensions, a null pointer or an
///         undersized workspace
///
/// @pre @p d_A, @p d_P and @p d_work must not overlap -- each Horner step is a
///      gemm, which requires its output distinct from both inputs.
/// @pre The handle is in default (host) pointer mode: the gemm scalars kOne and
///      kZero are passed by host address, like calaman.laqps / calaman.geqp3.
export template<typename T>
calaman::Status horner(wwr::wwrblasHandle_t handle, const int n, const T *d_coeffs,
                       const int degree, const T *d_A, const int lda, T *d_P, const int ldp,
                       void *d_work, const std::size_t work_bytes) {
  CLM_REQUIRE(n >= 1 && degree >= 0 && lda >= n && ldp >= n, wwr::WWRBLAS_STATUS_NOT_INITIALIZED);
  CLM_REQUIRE(all_nonnull(d_coeffs, d_A, d_P), wwr::WWRBLAS_STATUS_NOT_INITIALIZED);
  // Degree 0 is just c_0 * I: no product, so no scratch block.
  CLM_REQUIRE(degree == 0 || (d_work != nullptr && work_bytes >= horner_bufferSize<T>(n)),
              wwr::WWRBLAS_STATUS_NOT_INITIALIZED);

  // The diagonal-update launchers enqueue on the handle's own stream, the same
  // one the gemms run on, so the whole evaluation stays ordered on one stream.
  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(handle, &stream));

  // A gemm cannot write its own input, so the accumulator ping-pongs between the
  // caller's P and the scratch block. Each of the `degree` steps flips buffers,
  // so starting on the matching parity lands the last step in d_P.
  T *buf[2] = {d_P, static_cast<T *>(d_work)};
  const int ld[2] = {ldp, n};
  int cur = (degree % 2 == 0) ? 0 : 1;

  // P <- c_degree * I
  device::horner_set_scaled_identity(stream, buf[cur], n, ld[cur], d_coeffs + degree);

  for (int k = degree - 1; k >= 0; --k) {
    const int next = 1 - cur;

    // P_next <- P_cur * A
    CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &kOne<T>, buf[cur],
                         ld[cur], d_A, lda, &kZero<T>, buf[next], ld[next]));
    cur = next;

    // P <- P + c_k * I
    device::horner_add_scaled_identity(stream, buf[cur], n, ld[cur], d_coeffs + k);
  }

  return wwr::WWRBLAS_STATUS_SUCCESS;
}

extern template std::size_t horner_bufferSize<float>(int);
extern template std::size_t horner_bufferSize<double>(int);

extern template calaman::Status horner<float>(wwr::wwrblasHandle_t, int, const float *, int,
                                               const float *, int, float *, int, void *,
                                               std::size_t);
extern template calaman::Status horner<double>(wwr::wwrblasHandle_t, int, const double *, int,
                                               const double *, int, double *, int, void *,
                                               std::size_t);

} // namespace calaman
