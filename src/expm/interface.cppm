/**
 * @file interface.cppm
 * @brief Primary interface for calaman.expm -- the matrix exponential exp(A) by
 *        scaling and squaring with a diagonal Pade approximant (Higham's
 *        Algorithm 2.3)
 *
 * exp(A) for a square column-major A, by Higham's scaling-and-squaring method
 * (Higham, "The Scaling and Squaring Method for the Matrix Exponential
 * Revisited", SIAM J. Matrix Anal. Appl. 26(4), 2005):
 *
 *   1. Take the CHEAPEST degree m in {3, 5, 7, 9, 13} whose backward-error
 *      threshold theta_m already covers norm_1(A).
 *   2. Only if even theta_13 does not, SCALE: pick the smallest s with
 *      norm_1(A)/2^s <= theta_13, evaluate X = r_13(A/2^s), and square s times.
 *
 * NO BALANCING. expm does not balance the matrix itself: balancing is the
 * orthogonal similarity exp(A) = D exp(D^-1 A D) D^-1, which composes around this
 * routine rather than belonging inside it. A caller who wants it runs
 * calaman.gebal (GebalJob::Scale) to form D^-1 A D, calls expm on that, and
 * re-wraps the result with D and D^-1 (the README spells out the recipe).
 *
 * The [m/m] diagonal Pade approximant r_m(x) = q_m(x)^-1 p_m(x), with
 * q_m(x) = p_m(-x), matches exp(x) through order x^(2m). Splitting p_m into its
 * even and odd halves, p_m(A) = V + U and q_m(A) = V - U, so a single LU solve
 * of (V - U) X = (V + U) produces r_m(A). THE SIGN FLIP IS THE WHOLE TRICK: the
 * denominator costs nothing beyond the elementwise pass that already builds the
 * numerator -- a general polynomial evaluator (calaman.horner) cannot see that
 * relationship. The README has the degree-13 nested form and the cost table.
 *
 * STRUCTURE. expm is a HOST COMPOSITION over wrapped BLAS (gemm/geam), the
 * wrapped LU solve (getrf/getrs) and four fused kernels of
 * its own (expm.cu, reached through expm_bridge.h). The matrix products are the
 * OUTERMOST WarpWraps layer that does the job -- the type-safe wrappers in
 * wwr.wrappers.blas / wwr.wrappers.solver (CLAUDE.md, "prefer the outermost
 * layer"). It owns a launcher bridge (expm_bridge.h), a device-compiled unit
 * (expm.cu) and an explicit-instantiation unit (instantiations.cpp), the same
 * module/.cu split calaman.lacpy / calaman.gebal use.
 *
 * TEMPLATED OVER ALL FOUR element types (float, double, and the two complex
 * types), constrained by wwr::usual_fp. The Pade coefficients are real even for
 * a complex matrix, so the gemm scalars are the only complex constants needed;
 * there is no portable constexpr complex literal (calaman.common's constants.h
 * documents why), so as_element() builds them at run time from wwr.complex's
 * host constructors, which is legal in this module purview even though it is not
 * in a GMF.
 *
 * NO gemm3m. The reference this was ported from offered cublasGemm3m (Gauss's
 * 25%-fewer-flops complex product) behind an option; it is a cuBLAS-only call
 * with no hipBLAS counterpart (it lives in wwr.cuda.cublas_v2, not the neutral
 * layer), so routing through it would break the backend-neutrality this project
 * is built on. Every matrix product goes through the portable wwr::gemm.
 *
 * STATUS. Returns calaman::Status (calaman.error_handling), via CLM_TRY -- the
 * cross-domain convention the rest of src/ returns. A failed BLAS or solver call
 * carries its OWN domain's code (a getrf/getrs failure is a solver-domain status,
 * not the WWRBLAS_STATUS_INTERNAL_ERROR masquerade this once forced). The
 * genuinely host-side checks stay BLAS-domain outcome codes: a bad argument is
 * WWRBLAS_STATUS_INVALID_VALUE and an undersized workspace WWRBLAS_STATUS_ALLOC_FAILED,
 * both of which convert to Status implicitly.
 *
 * SYNCHRONIZES @p stream once, to read norm_1(A) onto the host (the degree and
 * the scaling exponent drive host-side control flow). Call pade() directly when
 * the norm is known a priori and a fully asynchronous path is required.
 *
 * Usage:
 *   import calaman.expm;
 *   import wwr.blas;     // wwrblasHandle_t, wwrblasCreate
 *   import wwr.solver;   // wwrsolverDnHandle_t, wwrsolverDnCreate
 *   import wwr.runtime_api;  // wwrStream_t
 *   // cublas, cusolver bound to the SAME stream; d_A, d_expA: n-by-n device
 *   std::size_t bytes = 0;
 *   calaman::expm_bufferSize<double>(cusolver, n, &bytes);
 *   // d_work: bytes of 256-aligned device scratch; d_info: device int[2]
 *   calaman::ExpmPlan plan{};
 *   calaman::expm<double>(cublas, cusolver, stream, n, d_A, n, d_expA, n,
 *                         d_work, bytes, d_info, &plan);
 */

module;

#include "expm_bridge.h"
// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.expm;

import std;
import wwr.runtime_api; // wwrStream_t, wwrError_t, wwrSuccess, wwrMemcpyAsync, wwrStreamSynchronize
import wwr.blas;    // wwrblasHandle_t, wwrblasStatus_t, WWRBLAS_*, pointer-mode get/set, GetStream
import wwr.solver;  // wwrsolverDnHandle_t, wwrsolverStatus_t, WWRSOLVER_STATUS_SUCCESS
import wwr.complex; // wwrFloatComplex, wwrDoubleComplex, make_wwr*Complex (host)
import wwr.wrappers.common;           // usual_fp, complex_fp, ComplexToRealType
import wwr.wrappers.blas;             // gemm, geam
import wwr.wrappers.solver;           // getrf, getrf_bufferSize, getrs (legacy, int-dimensioned)
import calaman.common;                // WorkspaceLayout, align_up
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// The fused kernels address elements with a flat int index, so n*n must fit in
// an int. floor(sqrt(INT_MAX)) = 46340; an n*n matrix that large is far past
// what a single device holds, but the bound is checked rather than assumed.
constexpr int kMaxDim = 46340;

/// @brief Build a real value as an element of T (complex types get a zero
///        imaginary part), the only complex constant the Pade evaluation needs.
///
/// Legal here, in the module purview, where wwr.complex is imported and its host
/// constructors are reachable -- unlike a GMF or a host constexpr, which cannot
/// see them (calaman.common's constants.h header documents the two walls).
template<wwr::usual_fp T>
T as_element(const wwr::ComplexToRealType<T> x) {
  if constexpr (std::is_same_v<T, wwr::wwrFloatComplex>) {
    return wwr::make_wwrFloatComplex(x, 0.0f);
  } else if constexpr (std::is_same_v<T, wwr::wwrDoubleComplex>) {
    return wwr::make_wwrDoubleComplex(x, 0.0);
  } else {
    return x;
  }
}

/**
 * @brief Device workspace slices for pade(), all 256-aligned.
 *
 * The layout is positional, and how many n*n blocks it spans depends on the
 * degree -- 3 at m = 3 and 6 at m = 9 and m = 13:
 *
 *   [P[0] | n*n T]  A^2 ... then holds U = A*W once V and W are formed
 *   [P[1] | n*n T]  A^4, when the degree needs it
 *   [P[2] | n*n T]  A^6,   "
 *   [P[3] | n*n T]  A^8 at m = 9; at m = 13 this slot is X, the nested inner
 *                   combination, used once per parity
 *   [V    | n*n T]  even part of the numerator ... then holds Q = V - U
 *   [W    | n*n T]  odd part, before multiplication by A
 *   [ipiv       | n int         ]
 *   [work_getrf | lwork_getrf T ]
 *
 * Two aliases keep the block count down. U reuses P[0]: the powers are all dead
 * by the time U is formed. Q reuses V: pade_split reads V(i) and writes Q(i)
 * from the same thread, so the denominator is built on top of the even half.
 */
template<wwr::usual_fp T>
struct PadeWorkspace {
  T *P[4] = {nullptr, nullptr, nullptr, nullptr};
  T *X = nullptr; // m = 13 only; aliases the P[3] slot
  T *V = nullptr;
  T *W = nullptr;
  T *U = nullptr; // aliases P[0] -- live only after V and W are formed
  T *Q = nullptr; // aliases V    -- live only after the split
  int *ipiv = nullptr;
  T *work_getrf = nullptr;

  /// @brief Carve the slices from @p layout, which sizes (null base) or carves
  ///        (real base) identically -- so pade_bufferSize and pade() share this
  ///        ONE region list and cannot drift. @p lwork_getrf is the getrf scratch
  ///        length, the one region whose size the caller has to query first.
  void carve(WorkspaceLayout &layout, const int n, const int m, const int lwork_getrf) {
    const std::size_t nn = static_cast<std::size_t>(n) * n;
    const int np = pade_num_powers(m);
    for (int k = 0; k < np; ++k) {
      P[k] = layout.fixed<T>(nn);
    }
    if (m == 13) {
      X = layout.fixed<T>(nn);
    }
    V = layout.fixed<T>(nn);
    W = layout.fixed<T>(nn);
    ipiv = layout.fixed<int>(static_cast<std::size_t>(n));
    work_getrf = layout.fixed<T>(static_cast<std::size_t>(lwork_getrf));
    U = P[0];
    Q = V;
  }
};

/**
 * @brief Device workspace slices for expm(), all 256-aligned.
 *
 *   [As     | n*n T                     ]  scaled matrix, and, once pade() has
 *                                          consumed it, the ping-pong target for
 *                                          the squaring phase -- hence the sq alias
 *   [colsum | n   ComplexToRealType<T>  ]  per-column absolute sums
 *   [pade   | PadeWorkspace, sized for the whole ladder ]
 */
template<wwr::usual_fp T>
struct ExpmWorkspace {
  using RealT = wwr::ComplexToRealType<T>;

  T *As = nullptr;
  T *sq = nullptr; // aliases As: only ever touched after pade() has returned
  RealT *colsum = nullptr;
  void *pade_base = nullptr;

  /// @brief Carve As and colsum from @p layout; pade_base is the layout cursor
  ///        after them -- where pade() carves its own sub-layout (expm sizes that
  ///        region for the worst degree, pade carves it for the chosen one).
  void carve(WorkspaceLayout &layout, const int n) {
    As = layout.fixed<T>(static_cast<std::size_t>(n) * n);
    sq = As;
    colsum = layout.fixed<RealT>(static_cast<std::size_t>(n));
    pade_base = layout.cursor();
  }
};

} // namespace calaman

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so each template carries its own
// `export` and the declarations below sit in the plain namespace.

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Backward-error thresholds
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// theta_m is the largest x with
//
//     sum_{k=2m+1}^{inf} |h_k| x^(k-1) <= u,     h(x) = log(exp(-x) r_m(x))
//
// Higham's backward-error criterion: r_m applied to the scaled matrix is the
// exact exponential of a nearby matrix A + E with norm(E) <= u * norm(A). The
// values come from evaluating that series in exact rational arithmetic and
// bisecting for the root at 60 significant digits (README has the derivation);
// at u = 2^-53 they reproduce Higham's published table exactly at m = 7, 9, 13.
//
//     m       u = 2^-53 (double)      u = 2^-24 (single)
//     3    1.495585217958292e-2    4.258730034897931e-1
//     5    2.539398330063232e-1    1.880152698533769
//     7    9.504178996162932e-1    3.925724846433284
//     9    2.097847961257067       6.249156334514102
//    13    5.371920351148152       1.124873763647540e1

/// @brief Backward-error threshold for degree @p m, or 0 if m is not on the ladder.
export template<wwr::usual_fp T>
constexpr wwr::ComplexToRealType<T> pade_theta(const int m) {
  using R = wwr::ComplexToRealType<T>;
  if constexpr (std::is_same_v<R, float>) {
    switch (m) {
    case 3:
      return 4.258730034897931e-1f;
    case 5:
      return 1.880152698533769f;
    case 7:
      return 3.925724846433284f;
    case 9:
      return 6.249156334514102f;
    case 13:
      return 1.124873763647540e1f;
    default:
      return R{0};
    }
  } else {
    switch (m) {
    case 3:
      return 1.495585217958292e-2;
    case 5:
      return 2.539398330063232e-1;
    case 7:
      return 9.504178996162932e-1;
    case 9:
      return 2.097847961257067;
    case 13:
      return 5.371920351148152;
    default:
      return R{0};
    }
  }
}

/// @brief The shape of one evaluation: which approximant, how many squarings, what it costs.
///
/// Both the prediction and the report: expm_plan() computes it from a norm ahead
/// of the call, and expm() fills it with the plan it executed. They are the same
/// type because expm is deterministic -- the degree and the scaling exponent
/// follow from the 1-norm alone, so what it WILL do and what it DID coincide, and
/// a second "info" struct would only restate a subset of this one.
export struct ExpmPlan {
  int m = 0;         ///< Pade degree chosen from {3, 5, 7, 9, 13}
  int s = 0;         ///< Squarings, i.e. the matrix is evaluated at A / 2^s
  int num_gemms = 0; ///< Matrix products the whole evaluation will issue
};

/**
 * @brief Pick the Pade degree and the scaling exponent for a given 1-norm.
 *
 * Higham's Algorithm 2.3: take the cheapest degree whose backward-error
 * threshold already covers the norm, and only fall back to scaling when even
 * theta_13 does not. Cost is pade_num_gemms(m) + s, and the ladder is ordered so
 * the first degree that fits is also the cheapest one that does. Exported
 * because it is the honest answer to "what will this cost me?" -- and the only
 * way a caller can predict the s that expm() reports.
 *
 * @param norm1 The matrix 1-norm; must be finite and non-negative.
 */
export template<wwr::usual_fp T>
ExpmPlan expm_plan(const wwr::ComplexToRealType<T> norm1) {
  using R = wwr::ComplexToRealType<T>;

  for (int i = 0; i < kNumPadeDegrees; ++i) {
    const int m = kPadeDegrees[i];
    if (norm1 <= pade_theta<T>(m)) {
      return ExpmPlan{m, 0, pade_num_gemms(m)};
    }
  }

  constexpr int m = 13;
  const R theta = pade_theta<T>(m);
  int s = static_cast<int>(std::ceil(std::log2(norm1 / theta)));
  if (s < 0) {
    s = 0;
  }
  return ExpmPlan{m, s, pade_num_gemms(m) + s};
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Matrix 1-norm
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * @brief Maximum absolute column sum of an n x n column-major matrix.
 *
 * Complex elements contribute their TRUE modulus, so this is the induced
 * 1-norm, not an |Re| + |Im| surrogate. The per-column sums land in @p d_colsum
 * and are then reduced in place.
 *
 * @tparam T       Element type (one of the four usual_fp types)
 * @param stream   Stream; work is enqueued here.
 * @param n        Matrix dimension.
 * @param d_A      Device matrix, column-major, leading dimension @p lda.
 * @param lda      Leading dimension of @p d_A (>= n).
 * @param d_colsum Device scratch, n reals (contents discarded).
 * @return The 1-norm, on the host. NaN if any element is NaN.
 *
 * @warning Synchronizes @p stream in order to return a host value.
 */
export template<wwr::usual_fp T>
wwr::ComplexToRealType<T> matrix_norm1(wwr::wwrStream_t stream, const int n, const T *d_A,
                                       const int lda, wwr::ComplexToRealType<T> *d_colsum) {
  using R = wwr::ComplexToRealType<T>;
  device::abs_colsums<T, R>(stream, n, d_A, lda, d_colsum);
  device::max_reduce<R>(stream, n, d_colsum);

  R host{};
  wwr::wwrMemcpyAsync(&host, d_colsum, sizeof(R), wwr::wwrMemcpyDeviceToHost, stream);
  wwr::wwrStreamSynchronize(stream);
  return host;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Workspace queries
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * @brief Device workspace, in bytes, required by pade() at degree @p m.
 *
 * Grows with the degree -- three n*n blocks at m = 3, six at m = 9 and m = 13 --
 * so a caller who knows its matrices are small-normed can ask for far less than
 * expm_bufferSize reserves.
 *
 * @param handle      Solver handle (queried for the getrf workspace size).
 * @param m           Pade degree; one of 3, 5, 7, 9, 13.
 * @param n           Matrix dimension.
 * @param lwork_bytes Output: required workspace in bytes.
 * @return Status: SUCCESS, WWRBLAS_STATUS_INVALID_VALUE for a bad degree or
 *         dimension, or the solver-domain status if the getrf query fails.
 */
export template<wwr::usual_fp T>
Status pade_bufferSize(wwr::wwrsolverDnHandle_t handle, const int m, const int n,
                       std::size_t *lwork_bytes) {
  if (pade_coeffs(m) == nullptr || n < 1) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  int lwork_getrf = 0;
  CLM_TRY(wwr::getrf_bufferSize<T>(handle, n, n, static_cast<T *>(nullptr), n, &lwork_getrf));

  WorkspaceLayout layout(nullptr); // null base: size only, from the same carve pade() runs
  PadeWorkspace<T> ws;
  ws.carve(layout, n, m, lwork_getrf);
  *lwork_bytes = layout.total();
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief Device workspace, in bytes, required by expm().
 *
 * The degree is chosen at run time from the matrix norm, so this reserves the
 * ladder's worst case: one n*n block for the scaled matrix, the six degree 13
 * needs, and a handful of O(n) vectors. Seven n*n blocks in total.
 *
 * @param handle      Solver handle (queried for the getrf workspace size).
 * @param n           Matrix dimension.
 * @param lwork_bytes Output: required workspace in bytes.
 */
export template<wwr::usual_fp T>
Status expm_bufferSize(wwr::wwrsolverDnHandle_t handle, const int n, std::size_t *lwork_bytes) {
  if (n < 1) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  int lwork_getrf = 0;
  CLM_TRY(wwr::getrf_bufferSize<T>(handle, n, n, static_cast<T *>(nullptr), n, &lwork_getrf));

  // As + colsum, then the pade region sized for the WORST degree on the ladder:
  // run the same PadeWorkspace carve over every degree and take the largest, so
  // expm() can carve any chosen degree into the region that follows.
  WorkspaceLayout layout(nullptr);
  ExpmWorkspace<T> ews;
  ews.carve(layout, n);

  std::size_t worst_pade = 0;
  for (int i = 0; i < kNumPadeDegrees; ++i) {
    WorkspaceLayout pade_layout(nullptr);
    PadeWorkspace<T> pws;
    pws.carve(pade_layout, n, kPadeDegrees[i], lwork_getrf);
    worst_pade = std::max(worst_pade, pade_layout.total());
  }

  *lwork_bytes = layout.total() + worst_pade;
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Diagonal Pade approximant, no scaling
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * @brief Evaluate the [m/m] diagonal Pade approximant r_m(A) = q_m(A)^-1 p_m(A).
 *
 * The unscaled kernel of the algorithm: accurate only while norm_1(A) stays
 * below pade_theta<T>(m); use expm() for a general matrix. Splits p_m into even
 * and odd halves so one LU solve of (V - U) X = (V + U) finishes the job (see
 * the file header and the README). Cost: pade_num_gemms(m) matrix products plus
 * one or two fused kernels, one getrf and one getrs. Unlike expm() it enqueues
 * work without ever synchronizing.
 *
 * @tparam T Element type (one of the four usual_fp types)
 * @param cublas_handle   BLAS handle, bound to @p stream, default (host) pointer mode.
 * @param cusolver_handle Solver handle, bound to @p stream.
 * @param stream          Stream the custom kernels are enqueued on.
 * @param m               Pade degree; one of 3, 5, 7, 9, 13.
 * @param n               Matrix dimension.
 * @param d_A             Input matrix, column-major, n x n, leading dimension @p lda.
 * @param lda             Leading dimension of @p d_A (>= n).
 * @param d_r             Output r_m(A), column-major, leading dimension @p ldr.
 * @param ldr             Leading dimension of @p d_r (>= n).
 * @param d_work          Device workspace of at least pade_bufferSize<T>(m, n) bytes.
 * @param lwork_bytes     Size of @p d_work in bytes.
 * @param d_info          Device array of 2 ints: [0] getrf info, [1] getrs info.
 * @return Status: SUCCESS, or the first error; a BLAS or solver failure carries
 *         its own domain's status, a bad argument WWRBLAS_STATUS_INVALID_VALUE,
 *         an undersized workspace WWRBLAS_STATUS_ALLOC_FAILED.
 *
 * @pre @p d_A and @p d_r must not overlap, and neither may overlap @p d_work.
 * @pre Both handles must be bound to @p stream.
 * @post The pointer mode of @p cublas_handle is left as it was found.
 */
export template<wwr::usual_fp T>
Status pade(wwr::wwrblasHandle_t cublas_handle, wwr::wwrsolverDnHandle_t cusolver_handle,
            wwr::wwrStream_t stream, const int m, const int n, const T *d_A, const int lda, T *d_r,
            const int ldr, void *d_work, const std::size_t lwork_bytes, int *d_info) {
  using RealT = wwr::ComplexToRealType<T>;

  const double *b = pade_coeffs(m);
  if (b == nullptr || n < 1 || n > kMaxDim || lda < n || ldr < n || d_work == nullptr) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  int lwork_getrf = 0;
  CLM_TRY(wwr::getrf_bufferSize<T>(cusolver_handle, n, n, static_cast<T *>(nullptr), n,
                                   &lwork_getrf));

  // One carve, sizing and laying out together: total() is the requirement, and
  // the pointers are unused until after the lwork check below passes.
  WorkspaceLayout layout(d_work);
  PadeWorkspace<T> ws;
  ws.carve(layout, n, m, lwork_getrf);
  if (lwork_bytes < layout.total()) {
    return wwr::WWRBLAS_STATUS_ALLOC_FAILED;
  }
  const int np = pade_num_powers(m);

  // The gemms use host scalars, so the handle must be in HOST pointer mode
  // regardless of how the caller left it; it is restored on every exit. CLM_TRY's
  // bail-out is a bare `return`, so the restore cannot live in a trailing lambda:
  // only a destructor fires on the early returns too, honoring the @post below.
  struct PointerModeGuard {
    wwr::wwrblasHandle_t h;
    wwr::wwrblasPointerMode_t saved;
    bool active;
    ~PointerModeGuard() {
      if (active) {
        wwr::wwrblasSetPointerMode(h, saved);
      }
    }
  };

  wwr::wwrblasPointerMode_t mode{};
  CLM_TRY(wwr::wwrblasGetPointerMode(cublas_handle, &mode));
  CLM_TRY(wwr::wwrblasSetPointerMode(cublas_handle, wwr::WWRBLAS_POINTER_MODE_HOST));
  const PointerModeGuard guard{cublas_handle, mode, true};

  const T one = as_element<T>(RealT{1});
  const T zero = as_element<T>(RealT{0});

  // Z = X * Y, and Z += X * Y (the beta = 1 that makes the degree-13 tail free)
  const auto matmul = [&](const T *X, const int ldx, const T *Y, const int ldy, T *Z,
                          const int ldz) {
    return wwr::gemm<T>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &one, X, ldx,
                        Y, ldy, &zero, Z, ldz);
  };
  const auto matmul_acc = [&](const T *X, const int ldx, const T *Y, const int ldy, T *Z,
                              const int ldz) {
    return wwr::gemm<T>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &one, X, ldx,
                        Y, ldy, &one, Z, ldz);
  };

  // ── Powers of A: A2, then A4, A6, A8 as far as the degree needs ────
  CLM_TRY(matmul(d_A, lda, d_A, lda, ws.P[0], n));
  if (np >= 2) {
    CLM_TRY(matmul(ws.P[0], n, ws.P[0], n, ws.P[1], n));
  }
  if (np >= 3) {
    CLM_TRY(matmul(ws.P[1], n, ws.P[0], n, ws.P[2], n));
  }
  if (np >= 4) {
    CLM_TRY(matmul(ws.P[1], n, ws.P[1], n, ws.P[3], n));
  }

  RealT cv[5] = {};
  RealT cw[5] = {};

  if (m != 13) {
    // ── Even and odd halves of the numerator, in one pass ──────────
    for (int i = 0; i <= np; ++i) {
      cv[i] = static_cast<RealT>(b[2 * i]);
      cw[i] = static_cast<RealT>(b[2 * i + 1]);
    }
    device::pade_even_odd<T, RealT>(stream, n, np, ws.P[0], ws.P[1], ws.P[2], ws.P[3], cv, cw, ws.V,
                                    ws.W);
  } else {
    // ── Odd parity: X = b9*A2 + b11*A4 + b13*A6, W = the tail ──────
    cv[0] = RealT{0};
    cv[1] = static_cast<RealT>(b[9]);
    cv[2] = static_cast<RealT>(b[11]);
    cv[3] = static_cast<RealT>(b[13]);
    cw[0] = static_cast<RealT>(b[1]);
    cw[1] = static_cast<RealT>(b[3]);
    cw[2] = static_cast<RealT>(b[5]);
    cw[3] = static_cast<RealT>(b[7]);
    device::pade_even_odd<T, RealT>(stream, n, 3, ws.P[0], ws.P[1], ws.P[2],
                                    static_cast<const T *>(nullptr), cv, cw, ws.X, ws.W);

    CLM_TRY(matmul_acc(ws.P[2], n, ws.X, n, ws.W, n)); // W += A6 * X

    // ── Even parity, reusing X now that the odd side is finished ───
    cv[0] = RealT{0};
    cv[1] = static_cast<RealT>(b[8]);
    cv[2] = static_cast<RealT>(b[10]);
    cv[3] = static_cast<RealT>(b[12]);
    cw[0] = static_cast<RealT>(b[0]);
    cw[1] = static_cast<RealT>(b[2]);
    cw[2] = static_cast<RealT>(b[4]);
    cw[3] = static_cast<RealT>(b[6]);
    device::pade_even_odd<T, RealT>(stream, n, 3, ws.P[0], ws.P[1], ws.P[2],
                                    static_cast<const T *>(nullptr), cv, cw, ws.X, ws.V);

    CLM_TRY(matmul_acc(ws.P[2], n, ws.X, n, ws.V, n)); // V += A6 * X
  }

  // ── U = A * W. Safe in place of P[0]: the powers are dead from here ─
  CLM_TRY(matmul(d_A, lda, ws.W, n, ws.U, n));

  // ── p(A) = V + U into the output, q(A) = V - U over V, in one pass ─
  device::pade_split<T>(stream, n, ws.U, ws.V, d_r, ldr, ws.Q);

  // ── Solve q(A) X = p(A) ────────────────────────────────────────────
  CLM_TRY(wwr::getrf<T>(cusolver_handle, n, n, ws.Q, n, ws.work_getrf, ws.ipiv, d_info));
  CLM_TRY(wwr::getrs<T>(cusolver_handle, wwr::WWRBLAS_OP_N, n, n, ws.Q, n, ws.ipiv, d_r, ldr,
                        d_info + 1));

  return wwr::WWRBLAS_STATUS_SUCCESS;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Scaling and squaring
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * @brief Matrix exponential by scaling and squaring with a diagonal Pade approximant.
 *
 * Higham's Algorithm 2.3 (see the file header). Total cost is
 * pade_num_gemms(m) + s matrix products; expm_plan() reports what a given norm
 * will cost before the call.
 *
 * @tparam T Element type (one of the four usual_fp types)
 * @param cublas_handle   BLAS handle, bound to @p stream, default (host) pointer mode.
 * @param cusolver_handle Solver handle, bound to @p stream.
 * @param stream          Stream all work is enqueued on.
 * @param n               Matrix dimension.
 * @param d_A             Input matrix, column-major, n x n, leading dimension @p lda.
 * @param lda             Leading dimension of @p d_A (>= n).
 * @param d_expA          Output exp(A), column-major, leading dimension @p lde.
 * @param lde             Leading dimension of @p d_expA (>= n).
 * @param d_work          Device workspace of at least expm_bufferSize<T>() bytes.
 * @param lwork_bytes     Size of @p d_work in bytes.
 * @param d_info          Device array of 2 ints: [0] getrf info, [1] getrs info.
 * @param plan            Optional output: the plan expm executed -- degree,
 *                        squarings and gemm count, the same ExpmPlan expm_plan()
 *                        would have predicted from norm_1(A).
 * @return Status: SUCCESS, or the first error; a BLAS or solver failure carries
 *         its own domain's status, a bad argument (including a 1-norm that is not
 *         finite) WWRBLAS_STATUS_INVALID_VALUE, an undersized workspace
 *         WWRBLAS_STATUS_ALLOC_FAILED.
 *
 * @pre @p d_A and @p d_expA must not overlap.
 * @pre Both handles must be bound to @p stream.
 * @post The pointer mode of @p cublas_handle is left as it was found.
 * @warning Synchronizes @p stream (see the file header).
 */
export template<wwr::usual_fp T>
Status expm(wwr::wwrblasHandle_t cublas_handle, wwr::wwrsolverDnHandle_t cusolver_handle,
            wwr::wwrStream_t stream, const int n, const T *d_A, const int lda, T *d_expA,
            const int lde, void *d_work, const std::size_t lwork_bytes, int *d_info,
            ExpmPlan *plan = nullptr) {
  using RealT = wwr::ComplexToRealType<T>;

  if (n < 1 || n > kMaxDim || lda < n || lde < n || d_work == nullptr) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  std::size_t required = 0;
  CLM_TRY(expm_bufferSize<T>(cusolver_handle, n, &required));
  if (lwork_bytes < required) {
    return wwr::WWRBLAS_STATUS_ALLOC_FAILED;
  }

  WorkspaceLayout layout(d_work);
  ExpmWorkspace<T> ws;
  ws.carve(layout, n);

  // HOST pointer mode for the host-scalar gemms/geams, restored on every exit.
  // A destructor, not a trailing lambda: CLM_TRY's bail-out is a bare `return`,
  // so only a guard's ~dtor restores on the early paths too (honoring the @post).
  struct PointerModeGuard {
    wwr::wwrblasHandle_t h;
    wwr::wwrblasPointerMode_t saved;
    bool active;
    ~PointerModeGuard() {
      if (active) {
        wwr::wwrblasSetPointerMode(h, saved);
      }
    }
  };

  wwr::wwrblasPointerMode_t mode{};
  CLM_TRY(wwr::wwrblasGetPointerMode(cublas_handle, &mode));
  CLM_TRY(wwr::wwrblasSetPointerMode(cublas_handle, wwr::WWRBLAS_POINTER_MODE_HOST));
  const PointerModeGuard guard{cublas_handle, mode, true};

  const T one = as_element<T>(RealT{1});
  const T zero = as_element<T>(RealT{0});

  // ── The norm the whole plan hangs off ──────────────────────────────
  const RealT norm = matrix_norm1<T>(stream, n, d_A, lda, ws.colsum);
  if (!std::isfinite(norm)) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  const ExpmPlan chosen = expm_plan<T>(norm);
  if (plan != nullptr) {
    *plan = chosen;
  }

  std::size_t pade_bytes = 0;
  CLM_TRY(pade_bufferSize<T>(cusolver_handle, chosen.m, n, &pade_bytes));

  // ── Pick what actually gets exponentiated ──────────────────────────
  const T *src = d_A;
  int ld_src = lda;

  if (chosen.s > 0) {
    const T factor = as_element<T>(std::ldexp(RealT{1}, -chosen.s));
    CLM_TRY(wwr::geam<T>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, &factor, d_A,
                         lda, &zero, d_A, lda, ws.As, n));
    src = ws.As;
    ld_src = n;
  }

  CLM_TRY(pade<T>(cublas_handle, cusolver_handle, stream, chosen.m, n, src, ld_src, d_expA, lde,
                  ws.pade_base, pade_bytes, d_info));

  // ── Square s times, ping-ponging between the output and the scratch ─
  //
  // ws.sq aliases ws.As, which pade() has finished reading by now. That is also
  // why the squaring cannot start in the scratch to save the copy below the way
  // horner() does: the scratch is still the input at the moment pade() would
  // write into it. One block of n^2 is worth more than one geam.
  T *cur = d_expA;
  int ld_cur = lde;
  T *other = ws.sq;
  int ld_other = n;

  for (int k = 0; k < chosen.s; ++k) {
    CLM_TRY(wwr::gemm<T>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &one, cur,
                         ld_cur, cur, ld_cur, &zero, other, ld_other));
    std::swap(cur, other);
    std::swap(ld_cur, ld_other);
  }

  // An odd number of squarings leaves the result in the scratch block.
  if (cur != d_expA) {
    CLM_TRY(wwr::geam<T>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, &one, cur,
                         ld_cur, &zero, cur, ld_cur, d_expA, lde));
  }

  return wwr::WWRBLAS_STATUS_SUCCESS;
}

// Paired with instantiations.cpp: each template is instantiated once inside this
// library (its body names the .cu-side launchers declared only in the GMF), so
// an importer never re-instantiates it. pade_theta and expm_plan are trivial
// constexpr/host helpers and are left to implicit instantiation.
extern template wwr::ComplexToRealType<float> matrix_norm1<float>(wwr::wwrStream_t, int,
                                                                  const float *, int, float *);
extern template wwr::ComplexToRealType<double> matrix_norm1<double>(wwr::wwrStream_t, int,
                                                                    const double *, int, double *);
extern template wwr::ComplexToRealType<wwr::wwrFloatComplex>
matrix_norm1<wwr::wwrFloatComplex>(wwr::wwrStream_t, int, const wwr::wwrFloatComplex *, int,
                                   float *);
extern template wwr::ComplexToRealType<wwr::wwrDoubleComplex>
matrix_norm1<wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, const wwr::wwrDoubleComplex *, int,
                                    double *);

extern template Status pade_bufferSize<float>(wwr::wwrsolverDnHandle_t, int, int, std::size_t *);
extern template Status pade_bufferSize<double>(wwr::wwrsolverDnHandle_t, int, int, std::size_t *);
extern template Status pade_bufferSize<wwr::wwrFloatComplex>(wwr::wwrsolverDnHandle_t, int, int,
                                                             std::size_t *);
extern template Status pade_bufferSize<wwr::wwrDoubleComplex>(wwr::wwrsolverDnHandle_t, int, int,
                                                              std::size_t *);

extern template Status expm_bufferSize<float>(wwr::wwrsolverDnHandle_t, int, std::size_t *);
extern template Status expm_bufferSize<double>(wwr::wwrsolverDnHandle_t, int, std::size_t *);
extern template Status expm_bufferSize<wwr::wwrFloatComplex>(wwr::wwrsolverDnHandle_t, int,
                                                             std::size_t *);
extern template Status expm_bufferSize<wwr::wwrDoubleComplex>(wwr::wwrsolverDnHandle_t, int,
                                                              std::size_t *);

extern template Status pade<float>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, wwr::wwrStream_t,
                                   int, int, const float *, int, float *, int, void *, std::size_t,
                                   int *);
extern template Status pade<double>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                    wwr::wwrStream_t, int, int, const double *, int, double *, int,
                                    void *, std::size_t, int *);
extern template Status pade<wwr::wwrFloatComplex>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                                  wwr::wwrStream_t, int, int,
                                                  const wwr::wwrFloatComplex *, int,
                                                  wwr::wwrFloatComplex *, int, void *, std::size_t,
                                                  int *);
extern template Status pade<wwr::wwrDoubleComplex>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                                   wwr::wwrStream_t, int, int,
                                                   const wwr::wwrDoubleComplex *, int,
                                                   wwr::wwrDoubleComplex *, int, void *,
                                                   std::size_t, int *);

extern template Status expm<float>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, wwr::wwrStream_t,
                                   int, const float *, int, float *, int, void *, std::size_t,
                                   int *, ExpmPlan *);
extern template Status expm<double>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                    wwr::wwrStream_t, int, const double *, int, double *, int,
                                    void *, std::size_t, int *, ExpmPlan *);
extern template Status expm<wwr::wwrFloatComplex>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                                  wwr::wwrStream_t, int,
                                                  const wwr::wwrFloatComplex *, int,
                                                  wwr::wwrFloatComplex *, int, void *, std::size_t,
                                                  int *, ExpmPlan *);
extern template Status expm<wwr::wwrDoubleComplex>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                                   wwr::wwrStream_t, int,
                                                   const wwr::wwrDoubleComplex *, int,
                                                   wwr::wwrDoubleComplex *, int, void *,
                                                   std::size_t, int *, ExpmPlan *);

} // namespace calaman
