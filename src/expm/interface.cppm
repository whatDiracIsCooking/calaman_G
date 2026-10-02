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
 *   1. Optionally BALANCE the matrix (calaman.gebal, scaling half only), keeping
 *      the result only if its 1-norm actually fell.
 *   2. Take the CHEAPEST degree m in {3, 5, 7, 9, 13} whose backward-error
 *      threshold theta_m already covers norm_1(A).
 *   3. Only if even theta_13 does not, SCALE: pick the smallest s with
 *      norm_1(A)/2^s <= theta_13, evaluate X = r_13(A/2^s), and square s times.
 *
 * The [m/m] diagonal Pade approximant r_m(x) = q_m(x)^-1 p_m(x), with
 * q_m(x) = p_m(-x), matches exp(x) through order x^(2m). Splitting p_m into its
 * even and odd halves, p_m(A) = V + U and q_m(A) = V - U, so a single LU solve
 * of (V - U) X = (V + U) produces r_m(A). THE SIGN FLIP IS THE WHOLE TRICK: the
 * denominator costs nothing beyond the elementwise pass that already builds the
 * numerator -- a general polynomial evaluator (calaman.horner) cannot see that
 * relationship. The README has the degree-13 nested form and the cost table.
 *
 * STRUCTURE. expm is a HOST COMPOSITION over wrapped BLAS (gemm/geam/scal/dgmm),
 * the wrapped LU solve (getrf/getrs), calaman.gebal and five fused kernels of
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
 * STATUS. Returns wwr::wwrblasStatus_t, like calaman.geqp3: the BLAS status enum
 * exposes neutral non-success codes (INVALID_VALUE, ALLOC_FAILED,
 * INTERNAL_ERROR) where the solver enum neutralizes only SUCCESS, so a mixed
 * BLAS+solver routine reports through the BLAS enum and a non-success solver
 * result surfaces as WWRBLAS_STATUS_INTERNAL_ERROR.
 *
 * SYNCHRONIZES @p stream: once to read norm_1(A) onto the host (the degree and
 * the scaling exponent drive host-side control flow), and twice more when
 * balancing is enabled (gebal is itself host-driven, and the balanced norm has
 * to be read back to decide whether to keep it). Call pade() directly when the
 * norm is known a priori and a fully asynchronous path is required.
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
 *   calaman::ExpmInfo info{};
 *   calaman::expm<double>(cublas, cusolver, stream, n, d_A, n, d_expA, n,
 *                         d_work, bytes, d_info, &info);
 */

module;

#include "expm_bridge.h"

export module calaman.expm;

import std;
import wwr.runtime_api;     // wwrStream_t, wwrError_t, wwrSuccess, wwrMemcpyAsync, wwrStreamSynchronize
import wwr.blas;            // wwrblasHandle_t, wwrblasStatus_t, WWRBLAS_*, pointer-mode get/set, GetStream
import wwr.solver;          // wwrsolverDnHandle_t, wwrsolverStatus_t, WWRSOLVER_STATUS_SUCCESS
import wwr.complex;         // wwrFloatComplex, wwrDoubleComplex, make_wwr*Complex (host)
import wwr.wrappers.common; // usual_fp, complex_fp, ComplexToRealType
import wwr.wrappers.blas;   // gemm, geam, scal, dgmm
import wwr.wrappers.solver; // getrf, getrf_bufferSize, getrs (legacy, int-dimensioned)
import calaman.common;      // WorkspaceBuilder, align_up
import calaman.gebal;       // gebal, gebal_bufferSize, GebalJob

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

/// @brief The largest pade_num_blocks over the whole ladder -- what expm() reserves.
constexpr int pade_max_blocks() {
  int best = 0;
  for (int i = 0; i < kNumPadeDegrees; ++i) {
    const int b = pade_num_blocks(kPadeDegrees[i]);
    if (b > best) {
      best = b;
    }
  }
  return best;
}

/**
 * @brief Device workspace slices for pade(), all 256-aligned.
 *
 * The layout is positional, and how many n*n blocks it spans depends on the
 * degree -- pade_num_blocks(m), 3 at m = 3 and 6 at m = 9 and m = 13:
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
  T *U = nullptr;    // aliases P[0] -- live only after V and W are formed
  T *Q = nullptr;    // aliases V    -- live only after the split
  int *ipiv = nullptr;
  T *work_getrf = nullptr;

  static PadeWorkspace make(void *d_work, const int n, const int m) {
    PadeWorkspace r;
    auto *p = static_cast<std::byte *>(d_work);

    const std::size_t mat =
        align_up(static_cast<std::size_t>(n) * n * sizeof(T), std::size_t{256});

    const int np = pade_num_powers(m);
    for (int k = 0; k < np; ++k) {
      r.P[k] = reinterpret_cast<T *>(p);
      p += mat;
    }
    if (m == 13) {
      r.X = reinterpret_cast<T *>(p);
      p += mat;
    }

    r.V = reinterpret_cast<T *>(p);
    p += mat;
    r.W = reinterpret_cast<T *>(p);
    p += mat;

    r.ipiv = reinterpret_cast<int *>(p);
    p += align_up(static_cast<std::size_t>(n) * sizeof(int), std::size_t{256});

    r.work_getrf = reinterpret_cast<T *>(p);

    r.U = r.P[0];
    r.Q = r.V;
    return r;
  }
};

/**
 * @brief Device workspace slices for expm(), all 256-aligned.
 *
 *   [As     | n*n T                     ]  scaled and/or balanced matrix, and,
 *                                          once pade() has consumed it, the
 *                                          ping-pong target for the squaring
 *                                          phase and the scratch for undoing the
 *                                          balancing -- hence the sq alias
 *   [colsum | n   ComplexToRealType<T>  ]  per-column absolute sums
 *   [scale  | n   ComplexToRealType<T>  ]  gebal's diagonal D
 *   [dscale | n   T                     ]  D widened to the element type
 *   [dinv   | n   T                     ]  D^-1, likewise
 *   [gwork  | gebal ints                ]  gebal scratch
 *   [pade   | PadeWorkspace, sized for the whole ladder ]
 */
template<wwr::usual_fp T>
struct ExpmWorkspace {
  using RealT = wwr::ComplexToRealType<T>;

  T *As = nullptr;
  T *sq = nullptr; // aliases As: only ever touched after pade() has returned
  RealT *colsum = nullptr;
  RealT *scale = nullptr;
  T *dscale = nullptr;
  T *dinv = nullptr;
  int *gwork = nullptr;
  void *pade_base = nullptr;

  static ExpmWorkspace make(void *d_work, const int n) {
    ExpmWorkspace r;
    auto *p = static_cast<std::byte *>(d_work);

    const auto bump = [&p](const std::size_t bytes) {
      auto *here = p;
      p += align_up(bytes, std::size_t{256});
      return here;
    };

    const std::size_t nn = static_cast<std::size_t>(n) * n;
    const std::size_t sn = static_cast<std::size_t>(n);

    r.As = reinterpret_cast<T *>(bump(nn * sizeof(T)));
    r.sq = r.As;

    r.colsum = reinterpret_cast<RealT *>(bump(sn * sizeof(RealT)));
    r.scale = reinterpret_cast<RealT *>(bump(sn * sizeof(RealT)));
    r.dscale = reinterpret_cast<T *>(bump(sn * sizeof(T)));
    r.dinv = reinterpret_cast<T *>(bump(sn * sizeof(T)));

    int gebal_ints = 0;
    gebal_bufferSize<T>(n, &gebal_ints);
    r.gwork = reinterpret_cast<int *>(bump(static_cast<std::size_t>(gebal_ints) * sizeof(int)));

    r.pade_base = p;
    return r;
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
export struct ExpmPlan {
  int m;         ///< Pade degree chosen from {3, 5, 7, 9, 13}
  int s;         ///< Squarings, i.e. the matrix is evaluated at A / 2^s
  int num_gemms; ///< Matrix products the whole evaluation will issue
};

/// @brief Whether expm() may balance the matrix before exponentiating it.
export enum class ExpmBalance {
  None, ///< Use the matrix exactly as given.
  Auto, ///< Balance, and keep the result only if the 1-norm actually fell.
};

/// @brief Tuning knobs for expm(); the default is the recommended setting.
export struct ExpmOptions {
  /// Diagonal balancing (calaman.gebal, scaling half only). Every factor of two
  /// it takes off the 1-norm is one squaring -- one n^3 product -- removed from
  /// the tail, and it improves accuracy on badly scaled matrices; the similarity
  /// is exact because gebal only scales by powers of two. Auto is safe: the
  /// balanced matrix is kept only when its 1-norm is strictly smaller, so the
  /// flop count can never rise. Set None to reproduce the unbalanced result bit
  /// for bit, or when Watkins' caveat applies ("A case where balancing is
  /// harmful", ETNA 2006).
  ExpmBalance balance = ExpmBalance::Auto;
};

/// @brief What expm() actually did, reported back on request.
export struct ExpmInfo {
  int s = 0;             ///< Squarings performed
  int m = 0;             ///< Pade degree used
  bool balanced = false; ///< Whether the balanced matrix was the one exponentiated
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
 * @return WWRBLAS_STATUS_SUCCESS, WWRBLAS_STATUS_INVALID_VALUE for a bad degree
 *         or dimension, or WWRBLAS_STATUS_INTERNAL_ERROR if the getrf query fails.
 */
export template<wwr::usual_fp T>
wwr::wwrblasStatus_t pade_bufferSize(wwr::wwrsolverDnHandle_t handle, const int m, const int n,
                                     std::size_t *lwork_bytes) {
  if (pade_coeffs(m) == nullptr || n < 1) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  int lwork_getrf = 0;
  if (wwr::getrf_bufferSize<T>(handle, n, n, static_cast<T *>(nullptr), n, &lwork_getrf) !=
      wwr::WWRSOLVER_STATUS_SUCCESS) {
    return wwr::WWRBLAS_STATUS_INTERNAL_ERROR;
  }

  WorkspaceBuilder ws;
  ws.add_fixed<T>(static_cast<std::size_t>(n) * n, static_cast<std::size_t>(pade_num_blocks(m)));
  ws.add_fixed<int>(static_cast<std::size_t>(n));            // ipiv
  ws.add_fixed<T>(static_cast<std::size_t>(lwork_getrf));    // getrf scratch
  *lwork_bytes = ws.total();
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
wwr::wwrblasStatus_t expm_bufferSize(wwr::wwrsolverDnHandle_t handle, const int n,
                                     std::size_t *lwork_bytes) {
  if (n < 1) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  int lwork_getrf = 0;
  if (wwr::getrf_bufferSize<T>(handle, n, n, static_cast<T *>(nullptr), n, &lwork_getrf) !=
      wwr::WWRSOLVER_STATUS_SUCCESS) {
    return wwr::WWRBLAS_STATUS_INTERNAL_ERROR;
  }

  int gebal_ints = 0;
  gebal_bufferSize<T>(n, &gebal_ints);

  const std::size_t sn = static_cast<std::size_t>(n);
  using RealT = wwr::ComplexToRealType<T>;

  WorkspaceBuilder ws;
  ws.add_fixed<T>(sn * sn);                                 // As (and, later, sq)
  ws.add_fixed<RealT>(sn, 2);                               // colsum, scale
  ws.add_fixed<T>(sn, 2);                                   // dscale, dinv
  ws.add_fixed<int>(static_cast<std::size_t>(gebal_ints));  // gebal scratch
  // The pade region, sized for the worst degree on the ladder.
  ws.add_fixed<T>(sn * sn, static_cast<std::size_t>(pade_max_blocks()));
  ws.add_fixed<int>(sn);                                    // ipiv
  ws.add_fixed<T>(static_cast<std::size_t>(lwork_getrf));   // getrf scratch

  *lwork_bytes = ws.total();
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
 * @return WWRBLAS_STATUS_SUCCESS, or the first error; a BLAS/solver failure is
 *         WWRBLAS_STATUS_INTERNAL_ERROR, a bad argument WWRBLAS_STATUS_INVALID_VALUE,
 *         an undersized workspace WWRBLAS_STATUS_ALLOC_FAILED.
 *
 * @pre @p d_A and @p d_r must not overlap, and neither may overlap @p d_work.
 * @pre Both handles must be bound to @p stream.
 * @post The pointer mode of @p cublas_handle is left as it was found.
 */
export template<wwr::usual_fp T>
wwr::wwrblasStatus_t pade(wwr::wwrblasHandle_t cublas_handle,
                          wwr::wwrsolverDnHandle_t cusolver_handle, wwr::wwrStream_t stream,
                          const int m, const int n, const T *d_A, const int lda, T *d_r,
                          const int ldr, void *d_work, const std::size_t lwork_bytes, int *d_info) {
  using RealT = wwr::ComplexToRealType<T>;

  const double *b = pade_coeffs(m);
  if (b == nullptr || n < 1 || n > kMaxDim || lda < n || ldr < n || d_work == nullptr) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  std::size_t required = 0;
  const wwr::wwrblasStatus_t qstatus = pade_bufferSize<T>(cusolver_handle, m, n, &required);
  if (qstatus != wwr::WWRBLAS_STATUS_SUCCESS) {
    return qstatus;
  }
  if (lwork_bytes < required) {
    return wwr::WWRBLAS_STATUS_ALLOC_FAILED;
  }

  const auto ws = PadeWorkspace<T>::make(d_work, n, m);
  const int np = pade_num_powers(m);

  // The gemms use host scalars, so the handle must be in HOST pointer mode
  // regardless of how the caller left it; it is restored on every exit.
  wwr::wwrblasPointerMode_t mode{};
  if (wwr::wwrblasGetPointerMode(cublas_handle, &mode) != wwr::WWRBLAS_STATUS_SUCCESS) {
    return wwr::WWRBLAS_STATUS_INTERNAL_ERROR;
  }
  if (wwr::wwrblasSetPointerMode(cublas_handle, wwr::WWRBLAS_POINTER_MODE_HOST) !=
      wwr::WWRBLAS_STATUS_SUCCESS) {
    return wwr::WWRBLAS_STATUS_INTERNAL_ERROR;
  }
  const auto restore = [&](const wwr::wwrblasStatus_t result) {
    wwr::wwrblasSetPointerMode(cublas_handle, mode);
    return result;
  };

  const T one = as_element<T>(RealT{1});
  const T zero = as_element<T>(RealT{0});

  // Z = X * Y, and Z += X * Y (the beta = 1 that makes the degree-13 tail free)
  const auto matmul = [&](const T *X, const int ldx, const T *Y, const int ldy, T *Z,
                          const int ldz) {
    return wwr::gemm<T>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &one, X, ldx, Y,
                        ldy, &zero, Z, ldz);
  };
  const auto matmul_acc = [&](const T *X, const int ldx, const T *Y, const int ldy, T *Z,
                              const int ldz) {
    return wwr::gemm<T>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &one, X, ldx, Y,
                        ldy, &one, Z, ldz);
  };

  // ── Powers of A: A2, then A4, A6, A8 as far as the degree needs ────
  if (matmul(d_A, lda, d_A, lda, ws.P[0], n) != wwr::WWRBLAS_STATUS_SUCCESS) {
    return restore(wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
  }
  if (np >= 2 && matmul(ws.P[0], n, ws.P[0], n, ws.P[1], n) != wwr::WWRBLAS_STATUS_SUCCESS) {
    return restore(wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
  }
  if (np >= 3 && matmul(ws.P[1], n, ws.P[0], n, ws.P[2], n) != wwr::WWRBLAS_STATUS_SUCCESS) {
    return restore(wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
  }
  if (np >= 4 && matmul(ws.P[1], n, ws.P[1], n, ws.P[3], n) != wwr::WWRBLAS_STATUS_SUCCESS) {
    return restore(wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
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

    if (matmul_acc(ws.P[2], n, ws.X, n, ws.W, n) != wwr::WWRBLAS_STATUS_SUCCESS) { // W += A6 * X
      return restore(wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
    }

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

    if (matmul_acc(ws.P[2], n, ws.X, n, ws.V, n) != wwr::WWRBLAS_STATUS_SUCCESS) { // V += A6 * X
      return restore(wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
    }
  }

  // ── U = A * W. Safe in place of P[0]: the powers are dead from here ─
  if (matmul(d_A, lda, ws.W, n, ws.U, n) != wwr::WWRBLAS_STATUS_SUCCESS) {
    return restore(wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
  }

  // ── p(A) = V + U into the output, q(A) = V - U over V, in one pass ─
  device::pade_split<T>(stream, n, ws.U, ws.V, d_r, ldr, ws.Q);

  // ── Solve q(A) X = p(A) ────────────────────────────────────────────
  if (wwr::getrf<T>(cusolver_handle, n, n, ws.Q, n, ws.work_getrf, ws.ipiv, d_info) !=
      wwr::WWRSOLVER_STATUS_SUCCESS) {
    return restore(wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
  }
  if (wwr::getrs<T>(cusolver_handle, wwr::WWRBLAS_OP_N, n, n, ws.Q, n, ws.ipiv, d_r, ldr,
                    d_info + 1) != wwr::WWRSOLVER_STATUS_SUCCESS) {
    return restore(wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
  }

  return restore(wwr::WWRBLAS_STATUS_SUCCESS);
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
 * @param info            Optional output: the degree, the squarings, whether
 *                        balancing was kept.
 * @param opts            Balancing setting; see ExpmOptions.
 * @return WWRBLAS_STATUS_SUCCESS, or the first error; a BLAS/solver failure is
 *         WWRBLAS_STATUS_INTERNAL_ERROR, a bad argument (including a 1-norm that
 *         is not finite) WWRBLAS_STATUS_INVALID_VALUE, an undersized workspace
 *         WWRBLAS_STATUS_ALLOC_FAILED.
 *
 * @pre @p d_A and @p d_expA must not overlap.
 * @pre Both handles must be bound to @p stream.
 * @post The pointer mode of @p cublas_handle is left as it was found.
 * @warning Synchronizes @p stream (see the file header).
 */
export template<wwr::usual_fp T>
wwr::wwrblasStatus_t expm(wwr::wwrblasHandle_t cublas_handle,
                          wwr::wwrsolverDnHandle_t cusolver_handle, wwr::wwrStream_t stream,
                          const int n, const T *d_A, const int lda, T *d_expA, const int lde,
                          void *d_work, const std::size_t lwork_bytes, int *d_info,
                          ExpmInfo *info = nullptr, const ExpmOptions opts = {}) {
  using RealT = wwr::ComplexToRealType<T>;

  if (n < 1 || n > kMaxDim || lda < n || lde < n || d_work == nullptr) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  std::size_t required = 0;
  const wwr::wwrblasStatus_t qstatus = expm_bufferSize<T>(cusolver_handle, n, &required);
  if (qstatus != wwr::WWRBLAS_STATUS_SUCCESS) {
    return qstatus;
  }
  if (lwork_bytes < required) {
    return wwr::WWRBLAS_STATUS_ALLOC_FAILED;
  }

  const auto ws = ExpmWorkspace<T>::make(d_work, n);

  wwr::wwrblasPointerMode_t mode{};
  if (wwr::wwrblasGetPointerMode(cublas_handle, &mode) != wwr::WWRBLAS_STATUS_SUCCESS) {
    return wwr::WWRBLAS_STATUS_INTERNAL_ERROR;
  }
  if (wwr::wwrblasSetPointerMode(cublas_handle, wwr::WWRBLAS_POINTER_MODE_HOST) !=
      wwr::WWRBLAS_STATUS_SUCCESS) {
    return wwr::WWRBLAS_STATUS_INTERNAL_ERROR;
  }
  const auto restore = [&](const wwr::wwrblasStatus_t result) {
    wwr::wwrblasSetPointerMode(cublas_handle, mode);
    return result;
  };

  const T one = as_element<T>(RealT{1});
  const T zero = as_element<T>(RealT{0});

  // ── The norm the whole plan hangs off ──────────────────────────────
  RealT norm = matrix_norm1<T>(stream, n, d_A, lda, ws.colsum);
  if (!std::isfinite(norm)) {
    return restore(wwr::WWRBLAS_STATUS_INVALID_VALUE);
  }

  // ── Balance, and keep it only if the norm actually fell ────────────
  //
  // Only the scaling half of gebal is run. The permutation half isolates
  // eigenvalues, which does nothing for the norm and would oblige us to undo a
  // permutation as well; with job = Scale, ilo = 1 and ihi = n, so the scale
  // vector is the diagonal of D over its whole length and the undo is two dgmms.
  bool balanced = false;
  if (opts.balance == ExpmBalance::Auto && n > 1 && norm > RealT{0}) {
    if (wwr::geam<T>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, &one, d_A, lda, &zero,
                     d_A, lda, ws.As, n) != wwr::WWRBLAS_STATUS_SUCCESS) {
      return restore(wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
    }

    int ilo = 1;
    int ihi = n;
    if (gebal<T>(stream, GebalJob::Scale, n, ws.As, n, &ilo, &ihi, ws.scale, ws.gwork) !=
        wwr::wwrSuccess) {
      return restore(wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
    }

    const RealT norm_b = matrix_norm1<T>(stream, n, ws.As, n, ws.colsum);
    if (std::isfinite(norm_b) && norm_b < norm) {
      balanced = true;
      norm = norm_b;
    }
  }

  const ExpmPlan plan = expm_plan<T>(norm);
  if (info != nullptr) {
    info->s = plan.s;
    info->m = plan.m;
    info->balanced = balanced;
  }

  std::size_t pade_bytes = 0;
  const wwr::wwrblasStatus_t pstatus = pade_bufferSize<T>(cusolver_handle, plan.m, n, &pade_bytes);
  if (pstatus != wwr::WWRBLAS_STATUS_SUCCESS) {
    return restore(pstatus);
  }

  // ── Pick what actually gets exponentiated ──────────────────────────
  const T *src = balanced ? ws.As : d_A;
  int ld_src = balanced ? n : lda;

  if (plan.s > 0) {
    const T factor = as_element<T>(std::ldexp(RealT{1}, -plan.s));
    if (balanced) {
      // As already holds the balanced matrix and is contiguous with ld = n, so
      // one scal over its n*n elements scales it in place.
      if (wwr::scal<T>(cublas_handle, n * n, &factor, ws.As, 1) != wwr::WWRBLAS_STATUS_SUCCESS) {
        return restore(wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
      }
    } else {
      if (wwr::geam<T>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, &factor, d_A, lda,
                       &zero, d_A, lda, ws.As, n) != wwr::WWRBLAS_STATUS_SUCCESS) {
        return restore(wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
      }
      src = ws.As;
      ld_src = n;
    }
  }

  const wwr::wwrblasStatus_t pade_status =
      pade<T>(cublas_handle, cusolver_handle, stream, plan.m, n, src, ld_src, d_expA, lde,
              ws.pade_base, pade_bytes, d_info);
  if (pade_status != wwr::WWRBLAS_STATUS_SUCCESS) {
    return restore(pade_status);
  }

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

  for (int k = 0; k < plan.s; ++k) {
    if (wwr::gemm<T>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &one, cur, ld_cur,
                     cur, ld_cur, &zero, other, ld_other) != wwr::WWRBLAS_STATUS_SUCCESS) {
      return restore(wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
    }
    std::swap(cur, other);
    std::swap(ld_cur, ld_other);
  }

  // An odd number of squarings leaves the result in the scratch block.
  if (cur != d_expA) {
    if (wwr::geam<T>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, &one, cur, ld_cur,
                     &zero, cur, ld_cur, d_expA, lde) != wwr::WWRBLAS_STATUS_SUCCESS) {
      return restore(wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
    }
  }

  // ── Undo the balancing: exp(A) = D exp(D^-1 A D) D^-1 ──────────────
  if (balanced) {
    device::expand_scale<T, RealT>(stream, n, ws.scale, ws.dscale, ws.dinv);

    if (wwr::dgmm<T>(cublas_handle, wwr::WWRBLAS_SIDE_LEFT, n, n, d_expA, lde, ws.dscale, 1, ws.sq,
                     n) != wwr::WWRBLAS_STATUS_SUCCESS) {
      return restore(wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
    }
    if (wwr::dgmm<T>(cublas_handle, wwr::WWRBLAS_SIDE_RIGHT, n, n, ws.sq, n, ws.dinv, 1, d_expA,
                     lde) != wwr::WWRBLAS_STATUS_SUCCESS) {
      return restore(wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
    }
  }

  return restore(wwr::WWRBLAS_STATUS_SUCCESS);
}

// Paired with instantiations.cpp: each template is instantiated once inside this
// library (its body names the .cu-side launchers declared only in the GMF), so
// an importer never re-instantiates it. pade_theta and expm_plan are trivial
// constexpr/host helpers and are left to implicit instantiation.
extern template wwr::ComplexToRealType<float>
matrix_norm1<float>(wwr::wwrStream_t, int, const float *, int, float *);
extern template wwr::ComplexToRealType<double>
matrix_norm1<double>(wwr::wwrStream_t, int, const double *, int, double *);
extern template wwr::ComplexToRealType<wwr::wwrFloatComplex>
matrix_norm1<wwr::wwrFloatComplex>(wwr::wwrStream_t, int, const wwr::wwrFloatComplex *, int,
                                   float *);
extern template wwr::ComplexToRealType<wwr::wwrDoubleComplex>
matrix_norm1<wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, const wwr::wwrDoubleComplex *, int,
                                    double *);

extern template wwr::wwrblasStatus_t pade_bufferSize<float>(wwr::wwrsolverDnHandle_t, int, int,
                                                            std::size_t *);
extern template wwr::wwrblasStatus_t pade_bufferSize<double>(wwr::wwrsolverDnHandle_t, int, int,
                                                             std::size_t *);
extern template wwr::wwrblasStatus_t
pade_bufferSize<wwr::wwrFloatComplex>(wwr::wwrsolverDnHandle_t, int, int, std::size_t *);
extern template wwr::wwrblasStatus_t
pade_bufferSize<wwr::wwrDoubleComplex>(wwr::wwrsolverDnHandle_t, int, int, std::size_t *);

extern template wwr::wwrblasStatus_t expm_bufferSize<float>(wwr::wwrsolverDnHandle_t, int,
                                                            std::size_t *);
extern template wwr::wwrblasStatus_t expm_bufferSize<double>(wwr::wwrsolverDnHandle_t, int,
                                                             std::size_t *);
extern template wwr::wwrblasStatus_t
expm_bufferSize<wwr::wwrFloatComplex>(wwr::wwrsolverDnHandle_t, int, std::size_t *);
extern template wwr::wwrblasStatus_t
expm_bufferSize<wwr::wwrDoubleComplex>(wwr::wwrsolverDnHandle_t, int, std::size_t *);

extern template wwr::wwrblasStatus_t pade<float>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                                 wwr::wwrStream_t, int, int, const float *, int,
                                                 float *, int, void *, std::size_t, int *);
extern template wwr::wwrblasStatus_t pade<double>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                                  wwr::wwrStream_t, int, int, const double *, int,
                                                  double *, int, void *, std::size_t, int *);
extern template wwr::wwrblasStatus_t
pade<wwr::wwrFloatComplex>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, wwr::wwrStream_t, int, int,
                           const wwr::wwrFloatComplex *, int, wwr::wwrFloatComplex *, int, void *,
                           std::size_t, int *);
extern template wwr::wwrblasStatus_t
pade<wwr::wwrDoubleComplex>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, wwr::wwrStream_t, int,
                            int, const wwr::wwrDoubleComplex *, int, wwr::wwrDoubleComplex *, int,
                            void *, std::size_t, int *);

extern template wwr::wwrblasStatus_t expm<float>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                                 wwr::wwrStream_t, int, const float *, int, float *,
                                                 int, void *, std::size_t, int *, ExpmInfo *,
                                                 ExpmOptions);
extern template wwr::wwrblasStatus_t expm<double>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                                  wwr::wwrStream_t, int, const double *, int,
                                                  double *, int, void *, std::size_t, int *,
                                                  ExpmInfo *, ExpmOptions);
extern template wwr::wwrblasStatus_t
expm<wwr::wwrFloatComplex>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, wwr::wwrStream_t, int,
                           const wwr::wwrFloatComplex *, int, wwr::wwrFloatComplex *, int, void *,
                           std::size_t, int *, ExpmInfo *, ExpmOptions);
extern template wwr::wwrblasStatus_t
expm<wwr::wwrDoubleComplex>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, wwr::wwrStream_t, int,
                            const wwr::wwrDoubleComplex *, int, wwr::wwrDoubleComplex *, int, void *,
                            std::size_t, int *, ExpmInfo *, ExpmOptions);

} // namespace calaman
