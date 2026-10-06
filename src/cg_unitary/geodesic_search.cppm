/**
 * @file geodesic_search.cppm
 * @brief The two geodesic line searches on U(n) -- paper Tables 1 and 2
 *
 * Partition of calaman.cg_unitary.
 *
 * Both searches rest on the same observation (paper §2.2). A geodesic on U(n) is
 * Gamma(mu) = exp(mu S) W with S skew-Hermitian, so S has purely imaginary
 * eigenvalues and exp(mu S) has eigenvalues exp(i omega_k mu): a smooth cost
 * function restricted to a geodesic is an *almost periodic* function of mu, and
 * so is its derivative. The highest frequency present is q |omega_max|, with q
 * the order of the cost function, so
 *
 *     T_mu = 2 pi / (q |omega_max|)                                 eq. (15)
 *
 * is an interval in which the derivative completes at most one cycle of its
 * fastest component -- and therefore crosses zero at most twice (§3.1). That
 * bound is what makes a low-order approximation of the derivative adequate, and
 * it is why the searches approximate the *derivative* and find its zeros rather
 * than approximating the cost and minimizing it.
 *
 * The derivative itself (eq. 14) needs no trace: with A = Psi(R W) the Euclidean
 * gradient at the trial point and B = H R W, the second factor of eq. (14) is
 * B^H, so dJhat/dmu = 2 sigma Re trace{A B^H} = 2 sigma Re dotc(B, A) over the
 * n^2 elements read as one vector, sigma = -1 when minimizing. One level-1
 * reduction per sample.
 *
 * ## What stays on the device
 *
 * The samples are taken in *device* pointer mode, so they never reach the host.
 * The Vandermonde solve, the windowed DFT, the root finding and the step
 * selection all run in single-block kernels (cg_unitary.cu) on those device
 * values. A search therefore costs O(1) host synchronizations -- one inside the
 * matrix exponential, which needs the matrix 1-norm on the host to pick its Pade
 * degree, and one to read the chosen step back for the solver's control flow --
 * rather than one per sample.
 */

module;

#include "cg_unitary_bridge.h"

export module calaman.cg_unitary:geodesic_search;

import std;
import wwr.blas;            // wwrblasHandle_t/Status, WWRBLAS_*, pointer mode
import wwr.solver;          // wwrsolverDnHandle_t
import wwr.runtime_api;     // wwrStream_t, wwrMemcpyAsync, wwrSuccess, wwrError_t
import wwr.complex;         // make_wwrFloatComplex / make_wwrDoubleComplex (host)
import wwr.wrappers.blas;   // gemm, gemv, dot, dotc, nrm2, scal, geam
import wwr.extension.blas;  // ScopedPointerMode
import calaman.common;      // kOne, kZero, real_fp, usual_fp, ComplexToRealType, RealToComplexType
import calaman.error_handling; // Status, PointerModeStatus
import calaman.expm;        // expm, matrix_norm1
import :buffer_size;
import :cost_function;

export namespace calaman {

/// @brief Which of the paper's two line searches to run.
enum class LineSearchMethod {
  /// Table 1: a low-order polynomial fit to the derivative over [0, T_mu),
  /// taking the first zero crossing. P + 1 gradient evaluations.
  Polynomial,
  /// Table 2: a Hann-windowed DFT of the derivative over N_T periods, taking the
  /// best of several minima. N_DFT gradient evaluations -- several times the
  /// cost, in exchange for large steps and a choice among minima (§3.2).
  Dft,
};

/// @brief Tuning for the line searches. The defaults are the paper's own (§5.1).
struct GeodesicSearchOptions {
  /// P in Table 1: the order of the approximating polynomial, 1 to 5.
  int poly_order = 5;
  /**
   * K in Table 2: samples per period T_mu. At least 2 by Nyquist; the paper uses
   * 3 to 5 and runs its own results at 3 (§5.1).
   *
   * 5 here, not 3, because of the sample cap. The paper's N_DFT is unbounded, but
   * the single-block root finder holds at most 31 samples, and at that cap K and
   * N_T trade against each other: K = 3, N_T = 10 spends the 31 samples on ten
   * periods (3.1 per period) and stalls around -19 dB of diagonality on the
   * Brockett criterion; K = 5 spends them on six (5.2 per period) and reaches the
   * exact maximum.
   */
  int dft_factor = 5;
  /// N_T in Table 2: how many periods T_mu the DFT interval spans. Capped in
  /// effect by the sample bound; see dft_factor.
  int dft_periods = 10;
  /**
   * Estimate |omega_max| by power iteration rather than the matrix 1-norm.
   *
   * ||H||_2 <= ||H||_1, so the surrogate over-estimates the frequency and
   * therefore *under*-estimates T_mu -- the safe direction: the interval shrinks
   * and the at-most-one-cycle bound still holds, at the cost of conservatively
   * short steps (up to a factor of sqrt(n)). The power iteration is O(n^2) per
   * step and recovers them.
   */
  bool use_spectral_radius = true;
};

/// @brief What a search did, reported back for tests and diagnostics.
template<calaman::usual_fp T>
struct GeodesicSearchInfo {
  using R = calaman::ComplexToRealType<T>;
  R omega_max = R(0);  ///< the frequency estimate that set the interval
  R interval = R(0);   ///< T_mu, or T_DFT for the DFT search
  int num_samples = 0; ///< derivative evaluations taken
  int num_roots = 0;   ///< candidates the root finder returned
  bool found = false;  ///< whether a usable step size came out
};

namespace cg_detail {

template<typename R>
constexpr R cg_two_pi() {
  return R(6.283185307179586476925286766559);
}

/// alpha as an element of T (complex types get a zero imaginary part). Legal in
/// the module purview, where wwr.complex's host constructors are reachable.
template<calaman::usual_fp T>
T cg_as_element(const calaman::ComplexToRealType<T> x) {
  if constexpr (std::is_same_v<T, wwr::wwrFloatComplex>) {
    return wwr::make_wwrFloatComplex(x, 0.0f);
  } else if constexpr (std::is_same_v<T, wwr::wwrDoubleComplex>) {
    return wwr::make_wwrDoubleComplex(x, 0.0);
  } else {
    return x;
  }
}

/// alpha * H into dst, with a real alpha widened to the element type.
template<calaman::usual_fp T>
wwr::wwrblasStatus_t scale_into(wwr::wwrblasHandle_t handle, const int n,
                                const calaman::ComplexToRealType<T> alpha, const T *d_H, const int ldh,
                                T *d_dst, const int ldd) {
  const T a = cg_as_element<T>(alpha);
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode guard{handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                                PointerModeStatus{&pm_status}};
  if (pm_status != wwr::WWRBLAS_STATUS_SUCCESS) {
    return pm_status;
  }
  return wwr::geam<T, int>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, &a, d_H, ldh,
                           &kZero<T>, d_H, ldh, d_dst, ldd);
}

/// G = alpha (X - X^H), the projection of X onto u(n) -- one geam.
template<calaman::usual_fp T>
wwr::wwrblasStatus_t skew_hermitian_part(wwr::wwrblasHandle_t handle, const int n,
                                         const calaman::ComplexToRealType<T> alpha, const T *d_X,
                                         const int ldx, T *d_G, const int ldg) {
  const T a = cg_as_element<T>(alpha);
  const T b = cg_as_element<T>(-alpha);
  const auto adj = calaman::real_fp<T> ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_C;
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode guard{handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                                PointerModeStatus{&pm_status}};
  if (pm_status != wwr::WWRBLAS_STATUS_SUCCESS) {
    return pm_status;
  }
  return wwr::geam<T, int>(handle, wwr::WWRBLAS_OP_N, adj, n, n, &a, d_X, ldx, &b, d_X, ldx, d_G,
                           ldg);
}

/// <A, A> as a Frobenius norm: nrm2 over the m*n packed elements (host result).
template<calaman::usual_fp T>
wwr::wwrblasStatus_t frobenius_norm(wwr::wwrblasHandle_t handle, const int m, const int n,
                                    const T *d_A, const int /*lda*/, calaman::ComplexToRealType<T> *out) {
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode guard{handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                                PointerModeStatus{&pm_status}};
  if (pm_status != wwr::WWRBLAS_STATUS_SUCCESS) {
    return pm_status;
  }
  return wwr::nrm2<T, int>(handle, m * n, d_A, 1, out);
}

/// Re trace{A^H B} as a host real. For complex T it is the real dot of the two
/// element arrays read as 2*m*n reals: sum(a.x b.x + a.y b.y) == Re dotc(A, B).
template<calaman::usual_fp T>
wwr::wwrblasStatus_t frobenius_dot_real(wwr::wwrblasHandle_t handle, const int m, const int n,
                                        const T *d_A, const int /*lda*/, const T *d_B,
                                        const int /*ldb*/, calaman::ComplexToRealType<T> *out) {
  using RealT = calaman::ComplexToRealType<T>;
  const int count = m * n;
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode guard{handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                                PointerModeStatus{&pm_status}};
  if (pm_status != wwr::WWRBLAS_STATUS_SUCCESS) {
    return pm_status;
  }
  if constexpr (calaman::real_fp<T>) {
    return wwr::dot<T, int>(handle, count, d_A, 1, d_B, 1, out);
  } else {
    const RealT *ar = reinterpret_cast<const RealT *>(d_A);
    const RealT *br = reinterpret_cast<const RealT *>(d_B);
    return wwr::dot<RealT, int>(handle, 2 * count, ar, 1, br, 1, out);
  }
}

/// One dot product into device memory: Re trace{A B^H} without leaving the GPU.
template<calaman::usual_fp T>
wwr::wwrblasStatus_t dot_to_device(wwr::wwrblasHandle_t handle, const int count, const T *d_x,
                                   const T *d_y, T *d_out) {
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode guard{handle, wwr::WWRBLAS_POINTER_MODE_DEVICE,
                                                PointerModeStatus{&pm_status}};
  if (pm_status != wwr::WWRBLAS_STATUS_SUCCESS) {
    return pm_status;
  }
  if constexpr (calaman::real_fp<T>) {
    return wwr::dot<T, int>(handle, count, d_x, 1, d_y, 1, d_out);
  } else {
    return wwr::dotc<T, int>(handle, count, d_x, 1, d_y, 1, d_out);
  }
}

/**
 * @brief |omega_max| of a skew-Hermitian H, by power iteration.
 *
 * Its eigenvalues are purely imaginary, so |omega_max| is the spectral norm
 * sigma_max(H) = sqrt(lambda_max(H^H H)). Power iteration on M = H^H H (Hermitian
 * PSD) drives v to the dominant eigenvector; the Rayleigh quotient v^H M v =
 * ||H v||^2 (with ||v|| = 1) converges to sigma_max^2. @p d_v is the warm start
 * (n), @p d_work two scratch vectors (2n: H v and M v). Writes *omega on the host.
 */
template<calaman::usual_fp T>
Status skew_spectral_radius(wwr::wwrblasHandle_t handle, wwr::wwrStream_t stream, const int n,
                            const T *d_H, const int ldh, T *d_v, T *d_work,
                            calaman::ComplexToRealType<T> *omega) {
  using RealT = calaman::ComplexToRealType<T>;
  *omega = RealT{0};
  if (n < 1) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
  T *y = d_work;
  T *z = d_work + n;
  const auto adj = calaman::real_fp<T> ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_C;

  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode guard{handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                                PointerModeStatus{&pm_status}};
  if (pm_status != wwr::WWRBLAS_STATUS_SUCCESS) {
    return pm_status;
  }

  // Seed v with ones when the warm-start vector is empty (first call reseeds it).
  RealT nv{0};
  wwr::wwrblasStatus_t bs = wwr::nrm2<T, int>(handle, n, d_v, 1, &nv);
  if (bs != wwr::WWRBLAS_STATUS_SUCCESS) {
    return bs;
  }
  if (!(nv > RealT{0})) {
    device::cg_seed_ones<T>(stream, d_v, n);
    bs = wwr::nrm2<T, int>(handle, n, d_v, 1, &nv);
    if (bs != wwr::WWRBLAS_STATUS_SUCCESS) {
      return bs;
    }
  }
  if (!(nv > RealT{0})) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
  RealT inv = RealT{1} / nv;
  bs = wwr::scal<T, int>(handle, n, &inv, d_v, 1);
  if (bs != wwr::WWRBLAS_STATUS_SUCCESS) {
    return bs;
  }

  RealT omega2{0};
  for (int it = 0; it < 100; ++it) {
    // y = H v, rho = ||H v||^2 (the Rayleigh quotient at the unit vector v).
    bs = wwr::gemv<T, int>(handle, wwr::WWRBLAS_OP_N, n, n, &kOne<T>, d_H, ldh, d_v, 1, &kZero<T>, y,
                           1);
    if (bs != wwr::WWRBLAS_STATUS_SUCCESS) {
      return bs;
    }
    RealT ny{0};
    bs = wwr::nrm2<T, int>(handle, n, y, 1, &ny);
    if (bs != wwr::WWRBLAS_STATUS_SUCCESS) {
      return bs;
    }
    const RealT rho = ny * ny;

    // z = H^H y = M v.
    bs = wwr::gemv<T, int>(handle, adj, n, n, &kOne<T>, d_H, ldh, y, 1, &kZero<T>, z, 1);
    if (bs != wwr::WWRBLAS_STATUS_SUCCESS) {
      return bs;
    }
    RealT nz{0};
    bs = wwr::nrm2<T, int>(handle, n, z, 1, &nz);
    if (bs != wwr::WWRBLAS_STATUS_SUCCESS) {
      return bs;
    }

    const bool converged =
        (omega2 > RealT{0}) && (std::abs(rho - omega2) <= RealT(1e-4) * rho);
    omega2 = rho;
    if (!(nz > RealT{0})) {
      break; // H is (numerically) zero on this subspace; no larger frequency
    }

    if (wwr::wwrMemcpyAsync(d_v, z, static_cast<std::size_t>(n) * sizeof(T),
                            wwr::wwrMemcpyDeviceToDevice, stream) != wwr::wwrSuccess) {
      return wwr::WWRBLAS_STATUS_EXECUTION_FAILED;
    }
    RealT invz = RealT{1} / nz;
    bs = wwr::scal<T, int>(handle, n, &invz, d_v, 1);
    if (bs != wwr::WWRBLAS_STATUS_SUCCESS) {
      return bs;
    }
    if (converged) {
      break;
    }
  }

  *omega = std::sqrt(omega2 > RealT{0} ? omega2 : RealT{0});
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// |omega_max|, by power iteration or by the 1-norm surrogate.
template<calaman::usual_fp T>
Status frequency_bound(wwr::wwrblasHandle_t cublas_handle, wwr::wwrStream_t stream, const int n,
                       const T *d_H, const int ldh, const CgSlices<T> &s,
                       const bool use_power_iteration, calaman::ComplexToRealType<T> *omega) {
  using RealT = calaman::ComplexToRealType<T>;
  if (use_power_iteration) {
    return skew_spectral_radius<T>(cublas_handle, stream, n, d_H, ldh, s.power_v, s.power_work,
                                   omega);
  }
  *omega = matrix_norm1<T>(stream, n, d_H, ldh, s.colsum);
  return std::isfinite(static_cast<double>(*omega)) ? Status{wwr::WWRBLAS_STATUS_SUCCESS}
                                                    : Status{wwr::WWRBLAS_STATUS_INTERNAL_ERROR};
}

/**
 * @brief The sampling sweep shared by both searches.
 *
 * Evaluates the geodesic derivative -- and, when @p want_cost, the cost itself --
 * at mu_i = i h for i = 0..num_samples-1, leaving both in device memory.
 *
 * Sample 0 is the current point, so it reuses the caller's W and its already
 * computed gradient: no exponential, no product, one gradient evaluation saved.
 */
template<calaman::usual_fp T, typename CostF>
  requires unitary_cost_function<CostF, T>
Status sample_derivative(wwr::wwrblasHandle_t cublas_handle, wwr::wwrsolverDnHandle_t cusolver_handle,
                         wwr::wwrStream_t stream, const int n, const T *d_W, const T *d_H,
                         const int ldh, const T *d_Psi, const CostF &cost,
                         const calaman::ComplexToRealType<T> step, const int num_samples,
                         const bool want_cost, const CgSlices<T> &s) {
  const std::size_t block_bytes = static_cast<std::size_t>(n) * n * sizeof(T);

  // R_1 = exp(step H); the sign of step carries the search direction.
  if (scale_into<T>(cublas_handle, n, step, d_H, ldh, s.tmp, n) != wwr::WWRBLAS_STATUS_SUCCESS) {
    return wwr::WWRBLAS_STATUS_INTERNAL_ERROR;
  }

  const Status ex = expm<T>(cublas_handle, cusolver_handle, stream, n, s.tmp, n, s.rot, n, s.scratch,
                            s.scratch_bytes, s.info);
  if (!ex.ok()) {
    return ex;
  }

  // Every product below takes host scalars; the derivative dot does not, and
  // restores the mode itself. The guard restores the caller's mode on every exit.
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode guard{cublas_handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                                PointerModeStatus{&pm_status}};
  if (pm_status != wwr::WWRBLAS_STATUS_SUCCESS) {
    return pm_status;
  }

  // R_i is advanced by repeated multiplication rather than re-exponentiated:
  // exp(i mu H) = [exp(mu H)]^i (§3.3). Hence one exponential per search. gemm
  // cannot write into one of its own inputs, so the accumulator and its target
  // swap each step.
  T *r_cur = s.rot_acc;
  T *r_next = s.rot_tmp;

  for (int i = 0; i < num_samples; ++i) {
    const T *w_trial = nullptr;
    const T *psi_trial = nullptr;

    if (i == 0) {
      // mu = 0 is the current point: no rotation, and the gradient the solver
      // already computed. One gradient evaluation saved per search.
      w_trial = d_W;
      psi_trial = d_Psi;
    } else {
      if (i == 1) {
        if (wwr::wwrMemcpyAsync(r_cur, s.rot, block_bytes, wwr::wwrMemcpyDeviceToDevice, stream) !=
            wwr::wwrSuccess) {
          return wwr::WWRBLAS_STATUS_EXECUTION_FAILED;
        }
      } else {
        if (wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &kOne<T>,
                              r_cur, n, s.rot, n, &kZero<T>, r_next, n) !=
            wwr::WWRBLAS_STATUS_SUCCESS) {
          return wwr::WWRBLAS_STATUS_INTERNAL_ERROR;
        }
        std::swap(r_cur, r_next);
      }

      // W_i = R_i W
      if (wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &kOne<T>,
                            r_cur, n, d_W, n, &kZero<T>, s.w_new, n) != wwr::WWRBLAS_STATUS_SUCCESS) {
        return wwr::WWRBLAS_STATUS_INTERNAL_ERROR;
      }

      if (cost.euclidean_gradient(cublas_handle, stream, n, s.w_new, n, s.psi_trial, n, s.scratch,
                                  s.scratch_bytes) != wwr::WWRBLAS_STATUS_SUCCESS) {
        return wwr::WWRBLAS_STATUS_INTERNAL_ERROR;
      }

      w_trial = s.w_new;
      psi_trial = s.psi_trial;

      // The cost functor is free to have changed the mode.
      wwr::wwrblasSetPointerMode(cublas_handle, wwr::WWRBLAS_POINTER_MODE_HOST);
    }

    // B = H W_i, the second factor of eq. (14) before conjugation.
    if (wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &kOne<T>,
                          d_H, ldh, w_trial, n, &kZero<T>, s.tmp, n) != wwr::WWRBLAS_STATUS_SUCCESS) {
      return wwr::WWRBLAS_STATUS_INTERNAL_ERROR;
    }

    // The sample itself: Re trace{Psi B^H} as one dot, straight to device memory.
    if (dot_to_device<T>(cublas_handle, n * n, s.tmp, psi_trial, s.dots + i) !=
        wwr::WWRBLAS_STATUS_SUCCESS) {
      return wwr::WWRBLAS_STATUS_INTERNAL_ERROR;
    }

    if (want_cost) {
      if (cost.value(cublas_handle, stream, n, w_trial, n, psi_trial, n, s.cost_dots + i, s.scratch,
                     s.scratch_bytes) != wwr::WWRBLAS_STATUS_SUCCESS) {
        return wwr::WWRBLAS_STATUS_INTERNAL_ERROR;
      }
      wwr::wwrblasSetPointerMode(cublas_handle, wwr::WWRBLAS_POINTER_MODE_HOST);
    }
  }

  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace cg_detail

/**
 * @brief Table 1: step size from a polynomial fit to the geodesic derivative.
 *
 * Samples the derivative at P + 1 equispaced points across [0, T_mu], fits a
 * degree-P polynomial through them, and takes its smallest positive real root --
 * the first zero crossing, hence the first local extremum along the geodesic
 * (§3.1). One matrix exponential; the remaining rotations are its powers.
 *
 * @param d_W   Current point on U(n); n x n, packed (leading dimension n).
 * @param d_H   Search direction in u(n), at the group identity.
 * @param d_Psi Euclidean gradient at @p d_W, packed. Reused as sample 0, which is
 *              why the solver passes the one it already has.
 * @param mu    Host out: the step size, or 0 when no usable root was found. A
 *              zero means "no extremum along this geodesic" and the caller must
 *              treat it as a failed search, not as a zero-length step.
 *
 * @warning Synchronizes @p stream twice: once inside the matrix exponential, and
 *          once to bring @p mu back for the caller's control flow.
 */
template<calaman::usual_fp T, typename CostF>
  requires unitary_cost_function<CostF, T>
Status geodesic_search_poly(wwr::wwrblasHandle_t cublas_handle,
                            wwr::wwrsolverDnHandle_t cusolver_handle, wwr::wwrStream_t stream,
                            const int n, const T *d_W, const T *d_H, const int ldh, const T *d_Psi,
                            const CostF &cost, const CgDirection dir, const CgSlices<T> &s,
                            const GeodesicSearchOptions &opts, calaman::ComplexToRealType<T> *mu,
                            GeodesicSearchInfo<T> *info = nullptr) {
  using RealT = calaman::ComplexToRealType<T>;
  *mu = RealT{0};
  if (info != nullptr) {
    *info = GeodesicSearchInfo<T>{};
  }

  const int P = opts.poly_order;
  if (P < 1 || P > 5) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  RealT omega{0};
  Status status =
      cg_detail::frequency_bound<T>(cublas_handle, stream, n, d_H, ldh, s, opts.use_spectral_radius,
                                    &omega);
  if (!status.ok()) {
    return status;
  }

  // A zero direction means the gradient has vanished; the caller reads mu = 0 as
  // a failed search and stops.
  if (!(omega > RealT{0})) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  const RealT t_mu = cg_detail::cg_two_pi<RealT>() / (static_cast<RealT>(CostF::order) * omega);
  const RealT h = t_mu / static_cast<RealT>(P);
  const RealT sgn = cg_sign<RealT>(dir);

  if (info != nullptr) {
    info->omega_max = omega;
    info->interval = t_mu;
    info->num_samples = P + 1;
  }

  status = cg_detail::sample_derivative<T, CostF>(cublas_handle, cusolver_handle, stream, n, d_W,
                                                  d_H, ldh, d_Psi, cost, sgn * h, P + 1,
                                                  /*want_cost=*/false, s);
  if (!status.ok()) {
    return status;
  }

  // Table 1 step 7: interpolate, on the device.
  device::cg_table1_coeffs<T, RealT>(stream, s.dots, P, h, RealT{2} * sgn, s.coeffs_real);

  // Table 1 step 8: the first zero crossing, also on the device. The bracket's
  // upper bound is T_mu -- a root beyond it would be extrapolation past the fit.
  device::cg_poly_smallest_positive_real_root<RealT>(stream, s.coeffs_real, P, t_mu, s.mu, s.found);

  RealT host_mu{0};
  int host_found = 0;
  wwr::wwrError_t e = wwr::wwrMemcpyAsync(&host_mu, s.mu, sizeof(RealT), wwr::wwrMemcpyDeviceToHost,
                                          stream);
  if (e == wwr::wwrSuccess) {
    e = wwr::wwrMemcpyAsync(&host_found, s.found, sizeof(int), wwr::wwrMemcpyDeviceToHost, stream);
  }
  if (e == wwr::wwrSuccess) {
    e = wwr::wwrStreamSynchronize(stream);
  }
  if (e != wwr::wwrSuccess) {
    return e;
  }

  const bool usable = (host_found != 0) && (host_mu > RealT{0}) && (host_mu <= t_mu);
  if (info != nullptr) {
    info->num_roots = host_found;
    info->found = usable;
  }
  *mu = usable ? host_mu : RealT{0};
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief Table 2: step size from a Hann-windowed DFT of the geodesic derivative.
 *
 * Samples the derivative across N_T periods T_mu, approximates the almost
 * periodic derivative by a genuinely periodic one via the DFT, and finds the
 * roots of the resulting Fourier polynomial that lie on the unit circle -- every
 * real step size at which the derivative vanishes inside the window. Several
 * minima are therefore available, and the best is taken (§3.2).
 *
 * More expensive than the polynomial search -- N_DFT gradient evaluations against
 * P + 1 -- and still one matrix exponential. Same argument and synchronization
 * contract as geodesic_search_poly.
 */
template<calaman::usual_fp T, typename CostF>
  requires unitary_cost_function<CostF, T>
Status geodesic_search_dft(wwr::wwrblasHandle_t cublas_handle,
                           wwr::wwrsolverDnHandle_t cusolver_handle, wwr::wwrStream_t stream,
                           const int n, const T *d_W, const T *d_H, const int ldh, const T *d_Psi,
                           const CostF &cost, const CgDirection dir, const CgSlices<T> &s,
                           const GeodesicSearchOptions &opts, calaman::ComplexToRealType<T> *mu,
                           GeodesicSearchInfo<T> *info = nullptr) {
  using RealT = calaman::ComplexToRealType<T>;
  using CplxT = calaman::RealToComplexType<RealT>;
  *mu = RealT{0};
  if (info != nullptr) {
    *info = GeodesicSearchInfo<T>{};
  }

  if (opts.dft_factor < 2 || opts.dft_periods < 1) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  RealT omega{0};
  Status status =
      cg_detail::frequency_bound<T>(cublas_handle, stream, n, d_H, ldh, s, opts.use_spectral_radius,
                                    &omega);
  if (!status.ok()) {
    return status;
  }
  if (!(omega > RealT{0})) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  const RealT t_mu = cg_detail::cg_two_pi<RealT>() / (static_cast<RealT>(CostF::order) * omega);

  // N_DFT odd, at least 2 samples per period T_mu by Nyquist, and inside the
  // single-block bound. The paper's K = 3, N_T = 10 gives exactly 31.
  int num_dft = 2 * ((opts.dft_factor * opts.dft_periods) / 2) + 1;
  if (num_dft > kCgMaxSamples - 1) {
    num_dft = ((kCgMaxSamples - 1) / 2) * 2 + 1;
  }
  if (num_dft < 3) {
    num_dft = 3;
  }

  // T_DFT follows from the sample count actually used, so a capped N_DFT shortens
  // the window rather than under-sampling it.
  const int periods = num_dft / opts.dft_factor;
  const RealT t_dft = static_cast<RealT>(periods > 0 ? periods : 1) * t_mu;
  const RealT h = t_dft / static_cast<RealT>(num_dft);
  const RealT sgn = cg_sign<RealT>(dir);

  if (info != nullptr) {
    info->omega_max = omega;
    info->interval = t_dft;
    info->num_samples = num_dft;
  }

  status = cg_detail::sample_derivative<T, CostF>(cublas_handle, cusolver_handle, stream, n, d_W,
                                                  d_H, ldh, d_Psi, cost, sgn * h, num_dft,
                                                  /*want_cost=*/true, s);
  if (!status.ok()) {
    return status;
  }

  // Steps 8 to 10: window and transform, on the device.
  device::cg_hann_dft<T, RealT, CplxT>(stream, s.dots, num_dft, RealT{2} * sgn, s.coeffs_cplx);

  // Step 11: the zero crossings of the reconstructed Fourier derivative, as
  // arguments theta = 2 pi mu / T_DFT -- found directly on the real trig
  // polynomial the centred coefficients define, no complex root solve.
  device::cg_dft_root_args<RealT, CplxT>(stream, s.coeffs_cplx, num_dft, s.args, s.num_args,
                                         s.root_info);

  // Step 13's inputs: the sampled cost values.
  device::cg_real_parts<T, RealT>(stream, s.cost_dots, num_dft, RealT{1}, s.cost_vals);

  // Steps 12 and 13: choose among the candidates, anchored on the sampled cost.
  device::cg_select_dft_step<RealT>(stream, s.args, s.num_args, s.cost_vals, num_dft, t_dft,
                                    dir == CgDirection::Maximize ? 1 : 0, s.mu);

  RealT host_mu{0};
  int host_num_args = 0;
  wwr::wwrError_t e = wwr::wwrMemcpyAsync(&host_mu, s.mu, sizeof(RealT), wwr::wwrMemcpyDeviceToHost,
                                          stream);
  if (e == wwr::wwrSuccess) {
    e = wwr::wwrMemcpyAsync(&host_num_args, s.num_args, sizeof(int), wwr::wwrMemcpyDeviceToHost,
                            stream);
  }
  if (e == wwr::wwrSuccess) {
    e = wwr::wwrStreamSynchronize(stream);
  }
  if (e != wwr::wwrSuccess) {
    return e;
  }

  if (host_mu > RealT{0}) {
    if (info != nullptr) {
      info->num_roots = host_num_args;
      info->found = true;
    }
    *mu = host_mu;
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  // No candidate survived. The paper sets mu_k = 0 and stops (Table 2 step 11),
  // but in a solver that is a stall, and it happens for a reason with a cheaper
  // remedy: the DFT window is sampled at only K points per period T_mu, so as the
  // gradient shrinks towards the optimum the reconstructed Fourier polynomial
  // degenerates and its roots drift off the unit circle. The polynomial search of
  // Table 1 samples the same geodesic an order of magnitude more finely over a
  // single period and is unaffected.
  //
  // So defer to it. The DFT method's advantage is large steps early (§3.2), not
  // local accuracy; falling back once it runs out makes this method a strict
  // improvement over that one rather than a gamble. The cost is one extra
  // exponential and P + 1 gradients, on this path only.
  return geodesic_search_poly<T, CostF>(cublas_handle, cusolver_handle, stream, n, d_W, d_H, ldh,
                                        d_Psi, cost, dir, s, opts, mu, info);
}

/// @brief Dispatch to whichever search @p method names.
template<calaman::usual_fp T, typename CostF>
  requires unitary_cost_function<CostF, T>
Status geodesic_search(wwr::wwrblasHandle_t cublas_handle, wwr::wwrsolverDnHandle_t cusolver_handle,
                       wwr::wwrStream_t stream, const int n, const T *d_W, const T *d_H,
                       const int ldh, const T *d_Psi, const CostF &cost, const CgDirection dir,
                       const LineSearchMethod method, const CgSlices<T> &s,
                       const GeodesicSearchOptions &opts, calaman::ComplexToRealType<T> *mu,
                       GeodesicSearchInfo<T> *info = nullptr) {
  if (method == LineSearchMethod::Dft) {
    return geodesic_search_dft<T, CostF>(cublas_handle, cusolver_handle, stream, n, d_W, d_H, ldh,
                                         d_Psi, cost, dir, s, opts, mu, info);
  }
  return geodesic_search_poly<T, CostF>(cublas_handle, cusolver_handle, stream, n, d_W, d_H, ldh,
                                        d_Psi, cost, dir, s, opts, mu, info);
}

} // namespace calaman
