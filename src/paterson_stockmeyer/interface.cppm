/**
 * @file interface.cppm
 * @brief Primary interface for calaman.paterson_stockmeyer -- matrix polynomial
 *        evaluation by the Paterson-Stockmeyer scheme
 *
 * Evaluates
 *
 *     p(A) = c_0 I + c_1 A + c_2 A^2 + ... + c_d A^d
 *
 * for a square column-major matrix A, in O(sqrt(d)) matrix products instead of
 * the d that Horner needs.
 *
 * The trick is to group the terms in blocks of s and treat the result as a
 * polynomial in X = A^s:
 *
 *     p(A) = sum_{j=0}^{r} B_j X^j,     X = A^s,   r = floor(d / s)
 *     B_j  = c_(js) I + c_(js+1) A + ... + c_(js+s-1) A^(s-1)
 *
 * Each B_j is a cheap linear combination of powers of A that are formed *once*
 * and shared by every block, so the only matrix products are
 *
 *     (s - 1)   to build A^2, A^3, ..., A^s
 *   + r         Horner steps over X
 *
 * Minimizing (s - 1) + floor(d / s) gives s ~ sqrt(d) and a total of roughly
 * 2*sqrt(d) - 1 products. For d = 8 that is 4 against Horner's 8; for d = 16,
 * 7 against 16. The saving is bought with memory: the power bank holds s - 1
 * matrices at once, where Horner holds one.
 *
 * Structurally the twin of calaman.horner, and its asymptotically-cheaper
 * sibling: a host wrapper over wrapped BLAS plus a device kernel, so it owns a
 * launcher bridge (paterson_stockmeyer_bridge.h), a device-compiled unit
 * (paterson_stockmeyer.cu) and an explicit-instantiation unit
 * (instantiations.cpp). Every matrix product is reached at the OUTERMOST
 * WarpWraps layer that does the job -- the type-safe dispatch wrapper in
 * wwr.wrappers.blas (CLAUDE.md, "prefer the outermost layer"). The one piece
 * that is genuinely per-element device work -- fusing each B_j from the powers
 * in a single pass -- is the bridge launcher.
 *
 * The coefficients live on the DEVICE, in ascending order: d_coeffs[k] is the
 * coefficient of A^k. Nothing in the evaluation reads them on the host -- the
 * block build (paterson_stockmeyer.cu, reached through
 * paterson_stockmeyer_bridge.h) takes the coefficients as a device pointer -- so
 * the whole call stays on the handle's stream and coefficients computed by a
 * previous kernel can be fed straight in. This is independent of the handle's
 * pointer mode: the gemm scalars are the host constants kOne/kZero from
 * calaman.common.
 *
 * Templated over float and double. Complex ?paterson_stockmeyer is a deliberate
 * later extension, for the reason calaman.common's constants.h documents: there
 * is no portable constexpr spelling of the kOne/kZero gemm scalars in complex --
 * the same wall calaman.horner stops at.
 *
 * @note On product count alone this never loses to Horner, at any degree -- at
 *       d = 1 it does not multiply at all. What calaman.horner still has is a
 *       workspace that does not depend on the degree: one scratch block, always,
 *       and no plan to reason about. It is the right choice for the low degrees
 *       that show up in Pade numerators and rational filters, and whenever extra
 *       n-by-n workspace is the binding constraint.
 *
 * Usage:
 *   import calaman.paterson_stockmeyer;
 *   import wwr.blas;          // wwrblasHandle_t, wwrblasCreate
 *   wwr::wwrblasHandle_t handle{};
 *   wwr::wwrblasCreate(&handle);          // default (host) pointer mode
 *   // d_coeffs: degree+1 device scalars; d_A, d_P: n-by-n device matrices
 *   const std::size_t bytes = calaman::paterson_stockmeyer_bufferSize<double>(n, degree);
 *   // d_work: bytes of 256-aligned device scratch
 *   calaman::paterson_stockmeyer<double>(handle, n, d_coeffs, degree, d_A, lda,
 *                                        d_P, ldp, d_work, bytes);
 */

module;

#include "paterson_stockmeyer_bridge.h"

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.paterson_stockmeyer;

import std;
import wwr.blas;          // wwrblasHandle_t, wwrblasStatus_t, WWRBLAS_OP_N, WWRBLAS_STATUS_*
import wwr.runtime_api;   // wwrStream_t (the type wwrblasGetStream writes)
import wwr.wrappers.blas; // gemm
import calaman.common;    // kZero<T> / kOne<T> (:constants), WorkspaceBuilder, align_up

// export import, not a plain import: paterson_stockmeyer RETURNS calaman::Status,
// so a consumer of `import calaman.paterson_stockmeyer;` must see Status's member
// functions, not just its name -- the same re-export diff_norm does.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so each entity below carries its own
// `export` and the declarations sit in the plain namespace.

// ========================================================================
// Choosing the block size
// ========================================================================

/// @brief Number of matrix products the scheme costs at block size s.
///
/// Exactly floor(degree / s) Horner steps over X = A^s, plus the products that
/// build the power bank. The bank normally has to reach X itself, costing s - 1
/// products -- but once s exceeds the degree there are no Horner steps left, the
/// polynomial is a single fused block, and the bank only has to reach A^degree.
///
/// That second regime is not a curiosity: at degree 2 it is the outright winner
/// (form A^2, then one fused block -- one product, against Horner's two), which
/// is why the minimization below scans all the way to s = degree + 1 rather than
/// stopping at s = degree.
export constexpr int paterson_stockmeyer_cost(const int degree, const int s) {
  if (s < 1 || degree < 1) {
    return 0;
  }

  const int r = degree / s;
  const int max_power = (r >= 1) ? s : degree;
  const int num_powers = (max_power >= 2) ? (max_power - 1) : 0;
  return num_powers + r;
}

/// @brief Cost-minimizing block size for a degree-d polynomial.
///
/// Scans s = 1 .. d+1 and takes the exact argmin, rather than rounding sqrt(d):
/// the degrees in play are small, so the scan is free, and it sidesteps both the
/// floor/ceil edge cases and the s > d regime that a sqrt heuristic misses
/// entirely. Ties go to the smallest s, which is the one with the smaller power
/// bank -- so a tie is broken toward less memory.
///
/// @return A block size in [1, degree + 1]; 1 only for degree 0, where there is
///         nothing to evaluate.
///
/// @note Degree 1 is the reason the guard below stops at degree 0 rather than
///       degree 1: p(A) = c_0 I + c_1 A is a single fused block with **no**
///       matrix product, which s = 2 expresses and s = 1 does not -- s = 1 would
///       spend a gemm forming (c_1 I) * A.
export constexpr int paterson_stockmeyer_block_size(const int degree) {
  if (degree < 1) {
    return 1;
  }

  int best_s = 1;
  int best_cost = paterson_stockmeyer_cost(degree, 1);
  for (int s = 2; s <= degree + 1; ++s) {
    const int cost = paterson_stockmeyer_cost(degree, s);
    if (cost < best_cost) {
      best_cost = cost;
      best_s = s;
    }
  }
  return best_s;
}

/// @brief The shape of one evaluation: block size, step count, workspace.
///
/// paterson_stockmeyer_bufferSize() and paterson_stockmeyer() both derive their
/// layout from this, so the size query and the evaluation cannot disagree. It is
/// exported because it is also the honest answer to "what will this cost me?" --
/// a caller weighing this against calaman.horner can read off both the product
/// count and the block count before committing.
export struct PatersonStockmeyerPlan {
  int s;           ///< Block size actually used
  int r;           ///< Horner steps over X = A^s, i.e. floor(degree / s)
  int num_powers;  ///< Powers held at once: A^2 .. A^(num_powers + 1)
  int num_blocks;  ///< Workspace blocks of n-by-n: num_powers, plus the accumulator
  int num_gemms;   ///< Matrix products the evaluation will issue
};

/// @brief Resolve the block size and the workspace layout for one evaluation.
///
/// @param degree      Polynomial degree, at least 0
/// @param s_requested Block size to force, or 0 to choose the cost-minimizing
///                    one. Values above degree + 1 are clamped -- there is no
///                    polynomial left to block beyond that.
export constexpr PatersonStockmeyerPlan paterson_stockmeyer_plan(const int degree,
                                                                 const int s_requested = 0) {
  PatersonStockmeyerPlan plan{1, 0, 0, 0, 0};
  if (degree < 0) {
    return plan;
  }

  int s = (s_requested > 0) ? s_requested : paterson_stockmeyer_block_size(degree);
  const int s_max = degree + 1;
  if (s > s_max) {
    s = s_max;
  }
  if (s < 1) {
    s = 1;
  }

  plan.s = s;
  plan.r = degree / s;

  // With no Horner steps the bank only has to reach A^degree; otherwise it must
  // reach X = A^s.
  const int max_power = (plan.r >= 1) ? s : degree;
  plan.num_powers = (max_power >= 2) ? (max_power - 1) : 0;

  // The accumulator ping-pongs against the caller's P, so it costs one block,
  // and only when there is a product to ping-pong across.
  plan.num_blocks = plan.num_powers + ((plan.r >= 1) ? 1 : 0);
  plan.num_gemms = plan.num_powers + plan.r;
  return plan;
}

// ========================================================================
// Workspace
// ========================================================================

/// @brief Device workspace required by paterson_stockmeyer(), in bytes.
///
/// The power bank plus one accumulator block, each n-by-n and 256-byte aligned,
/// and all live at once -- so they accumulate (add_fixed), not alias. Unlike
/// horner(), this grows with the degree -- that is the trade the scheme makes.
/// Use paterson_stockmeyer_plan() to see the block count directly.
///
/// @tparam T          Element type; one of the instantiated types (float, double)
/// @param n           Order of the matrix
/// @param degree      Polynomial degree
/// @param s_requested Block size to force, or 0 for the cost-minimizing choice.
///                    Must match the value passed to paterson_stockmeyer().
export template<typename T>
std::size_t paterson_stockmeyer_bufferSize(const int n, const int degree,
                                           const int s_requested = 0) {
  const std::size_t order = static_cast<std::size_t>(n < 1 ? 1 : n);
  const PatersonStockmeyerPlan plan = paterson_stockmeyer_plan(degree, s_requested);

  WorkspaceBuilder builder;
  for (int block = 0; block < plan.num_blocks; ++block) {
    builder.add_fixed<T>(order * order);
  }
  return builder.total();
}

/// @brief Elements between consecutive powers in the bank.
///
/// Each power is padded to the same 256-byte-aligned span the workspace builder
/// hands out, so every block in the bank is aligned, not just the first. The
/// division is exact for every supported T (4 and 8 bytes both divide 256).
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param n  Order of the matrix
export template<typename T>
std::size_t paterson_stockmeyer_power_stride(const int n) {
  const std::size_t order = static_cast<std::size_t>(n < 1 ? 1 : n);
  return align_up(order * order * sizeof(T), std::size_t{256}) / sizeof(T);
}

// ========================================================================
// Evaluation
// ========================================================================

/// @brief Matrix polynomial by Paterson-Stockmeyer: P = sum_{k=0}^{degree} c_k A^k
///
/// The coefficients are in ASCENDING order: d_coeffs[k] multiplies A^k, so the
/// array holds degree+1 entries. Identical convention, and identical results up
/// to rounding, to calaman.horner -- the two differ only in cost.
///
/// P is overwritten, not accumulated into. Only the n-by-n block of P is
/// written; the padding rows between n and @p ldp are left untouched, so P may
/// be a view into a larger allocation.
///
/// Each outer Horner step is a gemm, which cannot write one of its own inputs,
/// so the accumulator ping-pongs between the caller's P and the scratch block;
/// the starting buffer is chosen by the parity of r so the last step lands in
/// @p d_P and saves a final copy. Each block is written straight into the gemm's
/// output buffer, and the product is accumulated on top with beta = 1
/// (P_next <- P_cur * X + B_j), so there is one launch per step, not two.
///
/// The evaluation runs entirely on the handle's stream and never synchronizes:
/// the coefficients are read on the device by the block-build launcher, and the
/// gemm scalars are the host constants kOne/kZero.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle      GPU BLAS handle in default (host) pointer mode; its stream
///                    carries every operation and A, P, work live on its device
/// @param n           Order of A and P
/// @param d_coeffs    Device array of degree+1 coefficients, ascending in k
/// @param degree      Polynomial degree, at least 0
/// @param d_A         Input n-by-n column-major matrix
/// @param lda         Leading dimension of @p d_A; lda >= n
/// @param d_P         Output n-by-n column-major matrix, overwritten with p(A)
/// @param ldp         Leading dimension of @p d_P; ldp >= n
/// @param d_work      Device workspace, 256-byte aligned; may be null when the
///                    plan needs no blocks (degree 0)
/// @param work_bytes  Size of @p d_work; at least
///                    paterson_stockmeyer_bufferSize<T>(n, degree, s_requested)
/// @param s_requested Block size to force, or 0 for the cost-minimizing choice.
///                    Must match the value passed to the size query.
/// @return A successful Status, the first failing gemm status, or a Status
///         carrying WWRBLAS_STATUS_NOT_INITIALIZED for bad dimensions, a null
///         pointer or an undersized workspace (the only neutral non-success code
///         WarpWraps exposes, as calaman.diff_norm documents)
///
/// @pre @p d_A, @p d_P and @p d_work must not overlap -- each step is a gemm,
///      which requires its output distinct from both inputs.
/// @pre The handle is in default (host) pointer mode: the gemm scalars kOne and
///      kZero are passed by host address, like calaman.horner / calaman.laqps.
export template<typename T>
Status paterson_stockmeyer(wwr::wwrblasHandle_t handle, const int n, const T *d_coeffs,
                           const int degree, const T *d_A, const int lda, T *d_P, const int ldp,
                           void *d_work, const std::size_t work_bytes, const int s_requested = 0) {
  if (n < 1 || degree < 0 || lda < n || ldp < n || s_requested < 0) {
    return wwr::WWRBLAS_STATUS_NOT_INITIALIZED;
  }
  if (d_coeffs == nullptr || d_A == nullptr || d_P == nullptr) {
    return wwr::WWRBLAS_STATUS_NOT_INITIALIZED;
  }

  const PatersonStockmeyerPlan plan = paterson_stockmeyer_plan(degree, s_requested);
  if (plan.num_blocks > 0 &&
      (d_work == nullptr ||
       work_bytes < paterson_stockmeyer_bufferSize<T>(n, degree, s_requested))) {
    return wwr::WWRBLAS_STATUS_NOT_INITIALIZED;
  }

  // The block-build launcher enqueues on the handle's own stream, the same one
  // the gemms run on, so the whole evaluation stays ordered on one stream.
  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(handle, &stream));

  const std::size_t stride = paterson_stockmeyer_power_stride<T>(n);

  // A^2, A^3, ... A^(num_powers+1), then the accumulator's scratch half. Both
  // stay null when the plan needs no blocks, so no pointer is formed past the
  // end of a workspace the caller was allowed to omit.
  T *powers = (plan.num_powers > 0) ? static_cast<T *>(d_work) : nullptr;
  T *scratch = (plan.r >= 1)
                   ? static_cast<T *>(d_work) + static_cast<std::size_t>(plan.num_powers) * stride
                   : nullptr;

  // -- Build the power bank: A^k = A^(k-1) * A --------------------------------
  //
  // The first product reads A twice; the rest chain off the previous power.
  for (int k = 2; k <= plan.num_powers + 1; ++k) {
    const T *left = (k == 2) ? d_A : powers + static_cast<std::size_t>(k - 3) * stride;
    const int ld_left = (k == 2) ? lda : n;
    T *out = powers + static_cast<std::size_t>(k - 2) * stride;

    CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &kOne<T>, left,
                         ld_left, d_A, lda, &kZero<T>, out, n));
  }

  // X = A^s, the matrix the outer Horner runs over. For s == 1 that is A itself,
  // with the caller's leading dimension. When there are no Horner steps X does
  // not exist -- and A^s was never built -- so it stays null.
  const bool have_x = plan.r >= 1;
  const T *d_X = !have_x               ? nullptr
                 : (plan.s >= 2)       ? powers + static_cast<std::size_t>(plan.s - 2) * stride
                                       : d_A;
  const int ld_X = (plan.s >= 2) ? n : lda;

  // -- Outer Horner over X, one block per step --------------------------------
  //
  // A gemm cannot write one of its own inputs, so the accumulator ping-pongs
  // between the caller's P and the scratch block. Starting on the parity that
  // matches plan.r lands the last step in P and saves a final copy.
  T *buf[2] = {d_P, scratch};
  const int ld[2] = {ldp, n};
  int cur = (plan.r % 2 == 0) ? 0 : 1;

  // Terms in block j: s of them, except the last, which is whatever is left.
  const auto terms_in_block = [&](const int j) {
    const int remaining = degree - j * plan.s + 1;
    return remaining < plan.s ? remaining : plan.s;
  };

  // P <- B_r
  device::paterson_stockmeyer_build_block(stream, buf[cur], ld[cur], n, d_A, lda, powers, n, stride,
                                          d_coeffs + static_cast<std::size_t>(plan.r) * plan.s,
                                          terms_in_block(plan.r));

  for (int j = plan.r - 1; j >= 0; --j) {
    const int next = 1 - cur;

    // Write B_j into the destination first, then let the gemm accumulate
    // P_cur * X on top of it with beta = 1. One launch, not two.
    device::paterson_stockmeyer_build_block(stream, buf[next], ld[next], n, d_A, lda, powers, n,
                                            stride,
                                            d_coeffs + static_cast<std::size_t>(j) * plan.s,
                                            terms_in_block(j));

    CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &kOne<T>, buf[cur],
                         ld[cur], d_X, ld_X, &kOne<T>, buf[next], ld[next]));

    cur = next;
  }

  return wwr::WWRBLAS_STATUS_SUCCESS;
}

extern template std::size_t paterson_stockmeyer_bufferSize<float>(int, int, int);
extern template std::size_t paterson_stockmeyer_bufferSize<double>(int, int, int);

extern template std::size_t paterson_stockmeyer_power_stride<float>(int);
extern template std::size_t paterson_stockmeyer_power_stride<double>(int);

extern template Status paterson_stockmeyer<float>(wwr::wwrblasHandle_t, int, const float *, int,
                                                  const float *, int, float *, int, void *,
                                                  std::size_t, int);
extern template Status paterson_stockmeyer<double>(wwr::wwrblasHandle_t, int, const double *, int,
                                                   const double *, int, double *, int, void *,
                                                   std::size_t, int);

} // namespace calaman
