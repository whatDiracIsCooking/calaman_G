// cg_unitary.cu
//
// The device-kernel half of calaman.cg_unitary: the scalar stages of the two
// geodesic line searches (Abrudan, Eriksson & Koivunen, Signal Processing 89
// (2009), Tables 1 and 2) that no BLAS call expresses. Everything numeric in
// the solver -- the gradients, the geodesic rotations, the derivative samples --
// is a host composition of wrapped BLAS and calaman::expm; what stays here is
// the per-sample scalar work, run in single blocks on the device-resident dot
// products so a search synchronizes O(1) times:
//
//   * cg_table1_coeffs       : the Vandermonde (integer-matrix) solve, Table 1
//   * cg_poly_smallest_...   : its first positive root, by bracket + bisection
//   * cg_hann_dft            : the Hann-windowed centred DFT, Table 2 steps 8-10
//   * cg_dft_root_args       : zero crossings of the real trig polynomial the DFT
//                              coefficients define, bracketed on a grid, step 11
//   * cg_select_dft_step     : the cost-anchored step choice, steps 12-13
//   * cg_real_parts          : the sampled cost values, step 13's input
//   * cg_seed_ones           : the power-iteration warm start
//
// Shared unchanged between both backends, like nnls.cu: the .cu extension is all
// CMake needs (under HIP the CMakeLists forces -x hip), so no __CUDACC__ guard.
#include "cg_unitary_bridge.h"

#include "complex.h"

#include <cstddef>

namespace calaman::device {

namespace {

constexpr int kWarp = 32;

// ── real device math, overloaded on the real type ──────────────────────────
__device__ __forceinline__ float m_cos(const float x) { return cosf(x); }
__device__ __forceinline__ double m_cos(const double x) { return cos(x); }
__device__ __forceinline__ float m_sin(const float x) { return sinf(x); }
__device__ __forceinline__ double m_sin(const double x) { return sin(x); }
__device__ __forceinline__ float m_abs(const float x) { return fabsf(x); }
__device__ __forceinline__ double m_abs(const double x) { return fabs(x); }

template<typename R>
__device__ __forceinline__ R two_pi() {
  return R(6.283185307179586476925286766559);
}

// ── real part of a sample, whatever the element type ────────────────────────
// The derivative of eq. (14) and the cost values are both real parts of
// Frobenius inner products, so a complex sample contributes only its real part.
__device__ __forceinline__ float dot_real(const float v) { return v; }
__device__ __forceinline__ double dot_real(const double v) { return v; }
__device__ __forceinline__ float dot_real(const wwr::wwrFloatComplex v) { return wwr::wwrCrealf(v); }
__device__ __forceinline__ double dot_real(const wwr::wwrDoubleComplex v) { return wwr::wwrCreal(v); }

// ── one in the element type, for the power-iteration seed ───────────────────
__device__ __forceinline__ void set_one(float &x) { x = 1.0f; }
__device__ __forceinline__ void set_one(double &x) { x = 1.0; }
__device__ __forceinline__ void set_one(wwr::wwrFloatComplex &x) {
  x = wwr::make_wwrFloatComplex(1.0f, 0.0f);
}
__device__ __forceinline__ void set_one(wwr::wwrDoubleComplex &x) {
  x = wwr::make_wwrDoubleComplex(1.0, 0.0);
}

// ── complex accessors / constructors, overloaded on the complex type ────────
__device__ __forceinline__ float c_real(const wwr::wwrFloatComplex z) { return wwr::wwrCrealf(z); }
__device__ __forceinline__ double c_real(const wwr::wwrDoubleComplex z) { return wwr::wwrCreal(z); }
__device__ __forceinline__ float c_imag(const wwr::wwrFloatComplex z) { return wwr::wwrCimagf(z); }
__device__ __forceinline__ double c_imag(const wwr::wwrDoubleComplex z) { return wwr::wwrCimag(z); }
__device__ __forceinline__ wwr::wwrFloatComplex c_make(const float re, const float im) {
  return wwr::make_wwrFloatComplex(re, im);
}
__device__ __forceinline__ wwr::wwrDoubleComplex c_make(const double re, const double im) {
  return wwr::make_wwrDoubleComplex(re, im);
}

// ── seed ────────────────────────────────────────────────────────────────────

template<typename T>
__global__ void seed_ones_kernel(T *v, const int n) {
  for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
    set_one(v[i]);
  }
}

// ── Table 1: interpolate the derivative samples ─────────────────────────────

/// Solve M b = rhs for the integer matrix M[i][j] = i^j, then unscale by h^j.
/// One thread: the system is at most 5x5. Partial pivoting -- M is integer and
/// well conditioned but not diagonally dominant.
template<typename T, typename R>
__global__ void table1_coeffs_kernel(const T *d_dots, const int order, const R h,
                                     const R sample_scale, R *d_coeffs) {
  if (threadIdx.x != 0) {
    return;
  }

  R jp[kCgMaxSamples]; // Jp[i] = sample_scale * Re(dot_i)
  for (int i = 0; i <= order; ++i) {
    jp[i] = sample_scale * dot_real(d_dots[i]);
  }

  d_coeffs[0] = jp[0];
  if (order < 1) {
    return;
  }

  // M[i-1][j-1] = i^j for i, j = 1..order; rhs[i-1] = Jp[i] - Jp[0].
  R m[6][7];
  for (int i = 1; i <= order; ++i) {
    R p = R(1);
    for (int j = 1; j <= order; ++j) {
      p *= static_cast<R>(i);
      m[i - 1][j - 1] = p;
    }
    m[i - 1][order] = jp[i] - jp[0];
  }

  // Gaussian elimination with partial pivoting.
  for (int col = 0; col < order; ++col) {
    int piv = col;
    R best = m_abs(m[col][col]);
    for (int r = col + 1; r < order; ++r) {
      const R cand = m_abs(m[r][col]);
      if (cand > best) {
        best = cand;
        piv = r;
      }
    }
    if (!(best > R(0))) {
      // Singular: report a flat derivative so the caller falls back to mu = 0.
      for (int j = 1; j <= order; ++j) {
        d_coeffs[j] = R(0);
      }
      return;
    }
    if (piv != col) {
      for (int j = col; j <= order; ++j) {
        const R t = m[col][j];
        m[col][j] = m[piv][j];
        m[piv][j] = t;
      }
    }
    for (int r = col + 1; r < order; ++r) {
      const R factor = m[r][col] / m[col][col];
      for (int j = col; j <= order; ++j) {
        m[r][j] -= factor * m[col][j];
      }
    }
  }

  R b[6];
  for (int i = order - 1; i >= 0; --i) {
    R acc = m[i][order];
    for (int j = i + 1; j < order; ++j) {
      acc -= m[i][j] * b[j];
    }
    b[i] = acc / m[i][i];
  }

  // b_j = a_j h^j, so divide the powers of the sample spacing back out.
  R hp = R(1);
  for (int j = 1; j <= order; ++j) {
    hp *= h;
    d_coeffs[j] = (hp > R(0)) ? (b[j - 1] / hp) : R(0);
  }
}

/// Smallest positive real root of a degree-`order` polynomial in (0, upper],
/// by scanning for the first sign change on a fine grid and bisecting it.
template<typename R>
__global__ void poly_root_bracket_kernel(const R *d_coeffs, const int order, const R upper,
                                         R *d_mu, int *d_found) {
  if (threadIdx.x != 0) {
    return;
  }
  *d_mu = R(0);
  *d_found = 0;
  if (!(upper > R(0)) || order < 1) {
    return;
  }

  const auto eval = [&](const R x) {
    R acc = d_coeffs[order]; // Horner, descending from the top coefficient
    for (int j = order - 1; j >= 0; --j) {
      acc = acc * x + d_coeffs[j];
    }
    return acc;
  };

  constexpr int kGrid = 1024;
  const R step = upper / static_cast<R>(kGrid);
  R x_prev = R(0);
  R f_prev = eval(x_prev);
  // An exact zero at the origin is the current point, not a step: skip it.
  for (int i = 1; i <= kGrid; ++i) {
    const R x_cur = (i == kGrid) ? upper : static_cast<R>(i) * step;
    const R f_cur = eval(x_cur);
    const bool sign_change = (f_prev < R(0) && f_cur >= R(0)) || (f_prev > R(0) && f_cur <= R(0));
    if (sign_change) {
      // Bisection on [x_prev, x_cur]; f_prev and f_cur straddle zero.
      R lo = x_prev;
      R hi = x_cur;
      R flo = f_prev;
      for (int it = 0; it < 60; ++it) {
        const R mid = R(0.5) * (lo + hi);
        const R fmid = eval(mid);
        if ((flo < R(0) && fmid < R(0)) || (flo > R(0) && fmid > R(0))) {
          lo = mid;
          flo = fmid;
        } else {
          hi = mid;
        }
      }
      R root = R(0.5) * (lo + hi);
      if (root > R(0)) {
        *d_mu = root;
        *d_found = 1;
        return;
      }
    }
    x_prev = x_cur;
    f_prev = f_cur;
  }
}

// ── Table 2: Hann window and centred DFT ────────────────────────────────────

template<typename T, typename R, typename Cplx>
__global__ void hann_dft_kernel(const T *d_dots, const int num_samples, const R sample_scale,
                                Cplx *d_coeffs) {
  __shared__ R windowed[kCgMaxSamples];

  const int t = static_cast<int>(threadIdx.x);

  // D(mu_i) = h(i) Jhat'(mu_i), steps 8 and 9; the window is strictly positive
  // over this indexing, so it moves no zero of the derivative.
  if (t < num_samples) {
    const R jp = sample_scale * dot_real(d_dots[t]);
    const R w = R(0.5) - R(0.5) * m_cos(two_pi<R>() * static_cast<R>(t + 1) /
                                        static_cast<R>(num_samples + 1));
    windowed[t] = w * jp;
  }
  __syncthreads();

  // c_k for k = -(N-1)/2 .. (N-1)/2, written at index k + (N-1)/2. One thread
  // per coefficient, each an O(N) sum; N <= 31.
  if (t < num_samples) {
    const int half = (num_samples - 1) / 2;
    const int k = t - half;
    R re = R(0);
    R im = R(0);
    for (int i = 0; i < num_samples; ++i) {
      const R ang = -two_pi<R>() * static_cast<R>(i) * static_cast<R>(k) /
                    static_cast<R>(num_samples);
      re += windowed[i] * m_cos(ang);
      im += windowed[i] * m_sin(ang);
    }
    d_coeffs[t] = c_make(re, im);
  }
}

/// Zeros of the reconstructed Fourier derivative on the circle, as arguments.
///
/// The centred coefficients c_k (@p num_coeffs = N of them, index j <-> k =
/// j - (N-1)/2) define the trig polynomial D(theta) = sum_k c_k e^{i k theta},
/// which is real because the windowed derivative samples are real (so
/// c_{-k} = conj(c_k)). Its zeros over (0, 2 pi) are exactly the geodesic
/// derivative's zero crossings mapped to the angle theta = 2 pi mu / T_DFT -- the
/// step-size candidates. Found by bracketing sign changes on a fine grid and
/// bisecting, like the Table 1 root, so there is no complex root solve to
/// diverge and no leading-coefficient division to blow up. Arguments come out
/// ascending because the scan is.
template<typename R, typename Cplx>
__global__ void dft_root_args_kernel(const Cplx *d_coeffs, const int num_coeffs, R *d_args,
                                     int *d_num_args, int *d_info) {
  if (threadIdx.x != 0) {
    return;
  }
  *d_num_args = 0;
  *d_info = 0;
  const int N = num_coeffs;
  if (N < 2) {
    return;
  }
  const int half = (N - 1) / 2;

  const auto D = [&](const R theta) {
    R acc = R(0);
    for (int j = 0; j < N; ++j) {
      const R k = static_cast<R>(j - half);
      acc += c_real(d_coeffs[j]) * m_cos(k * theta) - c_imag(d_coeffs[j]) * m_sin(k * theta);
    }
    return acc;
  };

  constexpr int kGrid = 2048;
  const R tp = two_pi<R>();
  const R step = tp / static_cast<R>(kGrid);
  R x_prev = R(0);
  R f_prev = D(x_prev);
  int count = 0;
  for (int i = 1; i <= kGrid && count < kCgMaxSamples; ++i) {
    const R x = static_cast<R>(i) * step;
    const R f = D(x);
    const bool sign_change = (f_prev < R(0) && f >= R(0)) || (f_prev > R(0) && f <= R(0));
    if (sign_change && x > R(0)) {
      R lo = x_prev;
      R hi = x;
      R flo = f_prev;
      for (int it = 0; it < 50; ++it) {
        const R mid = R(0.5) * (lo + hi);
        const R fm = D(mid);
        if ((flo < R(0)) == (fm < R(0))) {
          lo = mid;
          flo = fm;
        } else {
          hi = mid;
        }
      }
      const R root = R(0.5) * (lo + hi);
      if (root > R(0) && root < tp) {
        d_args[count] = root;
        ++count;
      }
    }
    x_prev = x;
    f_prev = f;
  }
  *d_num_args = count;
}

// ── Table 2: choose among the candidate step sizes ──────────────────────────

template<typename R>
__global__ void select_dft_step_kernel(const R *d_args, const int *d_num_args, const R *d_cost,
                                       const int num_samples, const R t_dft, const int maximizing,
                                       R *d_mu) {
  if (threadIdx.x != 0) {
    return;
  }

  *d_mu = R(0);
  const int num_args = *d_num_args;
  if (num_args < 1 || num_samples < 1) {
    return;
  }

  // Step 13: the sample that best improves the cost. Anchoring on a measured
  // value rather than the parity of the crossing is what keeps a saddle point
  // inside the window from mislabelling every candidate.
  int best_i = 0;
  R best = d_cost[0];
  for (int i = 1; i < num_samples; ++i) {
    const bool better = maximizing ? (d_cost[i] > best) : (d_cost[i] < best);
    if (better) {
      best = d_cost[i];
      best_i = i;
    }
  }

  // best_i == 0 is not a failure: the window is coarse by design (§3.2), so once
  // the iterate sits near a sampled optimum every other sample reads worse while
  // a shorter step still improves. Falling through with mu_star = 0 selects the
  // smallest positive candidate -- the first extremum, the same answer Table 1
  // gives. Rejecting here would turn a sampling artefact into a hard stop.
  const R mu_star = static_cast<R>(best_i) * t_dft / static_cast<R>(num_samples);

  // Step 12: with a descent direction the derivative opens negative, so along
  // the ordered crossings the minima are the 1st, 3rd, 5th -- zero-based even.
  R chosen = R(0);
  R closest = R(-1);
  for (int l = 0; l < num_args; l += 2) {
    const R mu = d_args[l] * t_dft / two_pi<R>();
    if (!(mu > R(0))) {
      continue;
    }
    const R dist = m_abs(mu - mu_star);
    if (closest < R(0) || dist < closest) {
      closest = dist;
      chosen = mu;
    }
  }
  *d_mu = chosen;
}

// ── sampled cost values ──────────────────────────────────────────────────────

template<typename T, typename R>
__global__ void real_parts_kernel(const T *d_dots, const int count, const R scale, R *d_out) {
  const int t = static_cast<int>(threadIdx.x);
  if (t >= count) {
    return;
  }
  d_out[t] = scale * dot_real(d_dots[t]);
}

} // namespace

// ── host entry points ────────────────────────────────────────────────────────

template<typename T>
void cg_seed_ones(const wwr::wwrStream_t stream, T *const d_v, const int n) {
  if (n < 1) {
    return;
  }
  seed_ones_kernel<T><<<1, kWarp, 0, stream>>>(d_v, n);
}

template<typename T, typename R>
void cg_table1_coeffs(const wwr::wwrStream_t stream, const T *const d_dots, const int order,
                      const R h, const R sample_scale, R *const d_coeffs) {
  if (order < 0 || order > 5) {
    return;
  }
  table1_coeffs_kernel<T, R><<<1, 1, 0, stream>>>(d_dots, order, h, sample_scale, d_coeffs);
}

template<typename R>
void cg_poly_smallest_positive_real_root(const wwr::wwrStream_t stream, const R *const d_coeffs,
                                         const int order, const R upper, R *const d_mu,
                                         int *const d_found) {
  poly_root_bracket_kernel<R><<<1, 1, 0, stream>>>(d_coeffs, order, upper, d_mu, d_found);
}

template<typename T, typename R, typename Cplx>
void cg_hann_dft(const wwr::wwrStream_t stream, const T *const d_dots, const int num_samples,
                 const R sample_scale, Cplx *const d_coeffs) {
  if (num_samples < 1 || num_samples > kCgMaxSamples) {
    return;
  }
  hann_dft_kernel<T, R, Cplx><<<1, kWarp, 0, stream>>>(d_dots, num_samples, sample_scale, d_coeffs);
}

template<typename R, typename Cplx>
void cg_dft_root_args(const wwr::wwrStream_t stream, const Cplx *const d_coeffs,
                      const int num_coeffs, R *const d_args, int *const d_num_args,
                      int *const d_info) {
  if (num_coeffs < 2 || num_coeffs > kCgMaxSamples) {
    return;
  }
  dft_root_args_kernel<R, Cplx><<<1, 1, 0, stream>>>(d_coeffs, num_coeffs, d_args, d_num_args,
                                                     d_info);
}

template<typename R>
void cg_select_dft_step(const wwr::wwrStream_t stream, const R *const d_args,
                        const int *const d_num_args, const R *const d_cost, const int num_samples,
                        const R t_dft, const int maximizing, R *const d_mu) {
  if (num_samples < 1 || num_samples > kCgMaxSamples) {
    return;
  }
  select_dft_step_kernel<R><<<1, 1, 0, stream>>>(d_args, d_num_args, d_cost, num_samples, t_dft,
                                                 maximizing, d_mu);
}

template<typename T, typename R>
void cg_real_parts(const wwr::wwrStream_t stream, const T *const d_dots, const int count,
                   const R scale, R *const d_out) {
  if (count < 1 || count > kCgMaxSamples) {
    return;
  }
  real_parts_kernel<T, R><<<1, kWarp, 0, stream>>>(d_dots, count, scale, d_out);
}

// ── explicit instantiations: one per supported element type, matching the
//    bridge and the module's use sites ─────────────────────────────────────────

template void cg_seed_ones<float>(wwr::wwrStream_t, float *, int);
template void cg_seed_ones<double>(wwr::wwrStream_t, double *, int);
template void cg_seed_ones<wwr::wwrFloatComplex>(wwr::wwrStream_t, wwr::wwrFloatComplex *, int);
template void cg_seed_ones<wwr::wwrDoubleComplex>(wwr::wwrStream_t, wwr::wwrDoubleComplex *, int);

template void cg_table1_coeffs<float, float>(wwr::wwrStream_t, const float *, int, float, float,
                                             float *);
template void cg_table1_coeffs<double, double>(wwr::wwrStream_t, const double *, int, double, double,
                                               double *);
template void cg_table1_coeffs<wwr::wwrFloatComplex, float>(wwr::wwrStream_t,
                                                            const wwr::wwrFloatComplex *, int, float,
                                                            float, float *);
template void cg_table1_coeffs<wwr::wwrDoubleComplex, double>(wwr::wwrStream_t,
                                                              const wwr::wwrDoubleComplex *, int,
                                                              double, double, double *);

template void cg_poly_smallest_positive_real_root<float>(wwr::wwrStream_t, const float *, int, float,
                                                         float *, int *);
template void cg_poly_smallest_positive_real_root<double>(wwr::wwrStream_t, const double *, int,
                                                          double, double *, int *);

template void cg_hann_dft<float, float, wwr::wwrFloatComplex>(wwr::wwrStream_t, const float *, int,
                                                              float, wwr::wwrFloatComplex *);
template void cg_hann_dft<double, double, wwr::wwrDoubleComplex>(wwr::wwrStream_t, const double *,
                                                                 int, double,
                                                                 wwr::wwrDoubleComplex *);
template void
cg_hann_dft<wwr::wwrFloatComplex, float, wwr::wwrFloatComplex>(wwr::wwrStream_t,
                                                              const wwr::wwrFloatComplex *, int,
                                                              float, wwr::wwrFloatComplex *);
template void
cg_hann_dft<wwr::wwrDoubleComplex, double, wwr::wwrDoubleComplex>(wwr::wwrStream_t,
                                                                 const wwr::wwrDoubleComplex *, int,
                                                                 double, wwr::wwrDoubleComplex *);

template void cg_dft_root_args<float, wwr::wwrFloatComplex>(wwr::wwrStream_t,
                                                           const wwr::wwrFloatComplex *, int,
                                                           float *, int *, int *);
template void cg_dft_root_args<double, wwr::wwrDoubleComplex>(wwr::wwrStream_t,
                                                             const wwr::wwrDoubleComplex *, int,
                                                             double *, int *, int *);

template void cg_select_dft_step<float>(wwr::wwrStream_t, const float *, const int *, const float *,
                                        int, float, int, float *);
template void cg_select_dft_step<double>(wwr::wwrStream_t, const double *, const int *,
                                         const double *, int, double, int, double *);

template void cg_real_parts<float, float>(wwr::wwrStream_t, const float *, int, float, float *);
template void cg_real_parts<double, double>(wwr::wwrStream_t, const double *, int, double, double *);
template void cg_real_parts<wwr::wwrFloatComplex, float>(wwr::wwrStream_t,
                                                         const wwr::wwrFloatComplex *, int, float,
                                                         float *);
template void cg_real_parts<wwr::wwrDoubleComplex, double>(wwr::wwrStream_t,
                                                           const wwr::wwrDoubleComplex *, int,
                                                           double, double *);

} // namespace calaman::device
