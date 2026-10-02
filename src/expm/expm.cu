// expm.cu
//
// The device-kernel half of calaman.expm: the five genuinely per-element pieces
// of the scaling-and-squaring matrix exponential. Everything else in expm is a
// host composition of wrapped BLAS (gemm/geam/scal/dgmm), the wrapped LU solve
// (getrf/getrs) and calaman.gebal; these are the kernels those calls cannot
// express. Shared unchanged between both backends, like lacpy.cu / gebal.cu: a
// .cu is compiled by the backend compiler, so parallel_for's launch machinery,
// calaman.reduce_columns's segmented reduce, the raw <<<>>> launch syntax and
// the complex types/accessors (through complex.h) are all available directly.
//
// NEUTRAL COMPLEX, NEVER .x/.y. cuFloatComplex is an operator-less float2 whose
// components are reached as .x/.y, but hipFloatComplex is a class -- so raw
// field access is not portable. The real/imag components are read through
// complex.h's wwrCreal*/wwrCimag* accessors, the true modulus through
// wwrCabs*/wwrCabsf, and a new value built with make_wwr*Complex; the
// per-precision spelling lives in the elem_ops specializations below, matching
// gebal.cu's elem_ops and complex_cast.cu's complex_ops.
//
// The Pade coefficients are REAL even when the matrix is complex, so the only
// mixed operation the fused kernels need is "accumulate a real multiple of an
// element" (fma_r) and "add a real to the diagonal" (add_r).
#include "expm_bridge.h"

#include "complex.h"
#include "extension/parallel_for/parallel_for.cuh"
#include "reduce_columns/reduce_columns.cuh"

#include <cmath>
#include <cstddef>
#include <utility>

namespace calaman::device {

// complex.h puts the neutral complex types in namespace wwr; pull the two type
// names in so the elem_ops specializations and the explicit instantiations below
// can spell them bare. The accessor / constructor calls stay wwr::-qualified.
using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

namespace {

// ── element operations, one per expm element type ───────────────────────────
//
// Primary template covers the real types (float, double); the two complex
// specializations route every component access through the wwrC* accessors.

template<typename T>
struct elem_ops {
  using real_type = T;
  static __device__ T zero() { return T(0); }
  static __device__ T from_r(T c) { return c; }
  static __device__ T fma_r(T c, T x, T acc) { return acc + c * x; }
  static __device__ T add_r(T a, T c) { return a + c; }
  static __device__ T add(T a, T b) { return a + b; }
  static __device__ T sub(T a, T b) { return a - b; }
  static __device__ T modulus(T a) { return a < T(0) ? -a : a; }
};

template<>
struct elem_ops<wwrFloatComplex> {
  using T = wwrFloatComplex;
  using real_type = float;
  static __device__ T zero() { return wwr::make_wwrFloatComplex(0.0f, 0.0f); }
  static __device__ T from_r(float c) { return wwr::make_wwrFloatComplex(c, 0.0f); }
  static __device__ T fma_r(float c, T x, T acc) {
    return wwr::make_wwrFloatComplex(wwr::wwrCrealf(acc) + c * wwr::wwrCrealf(x),
                                     wwr::wwrCimagf(acc) + c * wwr::wwrCimagf(x));
  }
  static __device__ T add_r(T a, float c) {
    return wwr::make_wwrFloatComplex(wwr::wwrCrealf(a) + c, wwr::wwrCimagf(a));
  }
  static __device__ T add(T a, T b) { return wwr::wwrCaddf(a, b); }
  static __device__ T sub(T a, T b) { return wwr::wwrCsubf(a, b); }
  static __device__ float modulus(T a) { return wwr::wwrCabsf(a); }
};

template<>
struct elem_ops<wwrDoubleComplex> {
  using T = wwrDoubleComplex;
  using real_type = double;
  static __device__ T zero() { return wwr::make_wwrDoubleComplex(0.0, 0.0); }
  static __device__ T from_r(double c) { return wwr::make_wwrDoubleComplex(c, 0.0); }
  static __device__ T fma_r(double c, T x, T acc) {
    return wwr::make_wwrDoubleComplex(wwr::wwrCreal(acc) + c * wwr::wwrCreal(x),
                                      wwr::wwrCimag(acc) + c * wwr::wwrCimag(x));
  }
  static __device__ T add_r(T a, double c) {
    return wwr::make_wwrDoubleComplex(wwr::wwrCreal(a) + c, wwr::wwrCimag(a));
  }
  static __device__ T add(T a, T b) { return wwr::wwrCadd(a, b); }
  static __device__ T sub(T a, T b) { return wwr::wwrCsub(a, b); }
  static __device__ double modulus(T a) { return wwr::wwrCabs(a); }
};

// ── V = cv[0]*I + sum_k cv[k+1]*P_k,  W = cw[0]*I + sum_k cw[k+1]*P_k ─────────
//
// NP is a template parameter so the accumulation unrolls and the power pointers
// land in registers; the launcher below dispatches on the runtime count. All
// members are const (device_functor forbids a mutable member -- the grid-constant
// kernel copy would make a write to one UB), and the coefficient arrays are
// carried BY VALUE: a host array is not addressable from device code.
template<typename T, int NP>
struct PadeEvenOddFunctor {
  using ops = elem_ops<T>;
  using R = typename ops::real_type;

  // The pointer/coefficient arrays are plain (non-const) members so the struct
  // is unambiguously trivially copyable; the const scalar members below are what
  // delete copy-assignment, which is the property device_functor actually needs
  // (a const class- or array-type member is the shape its note warns against).
  const T *d_P_[NP];
  T *const d_V_;
  T *const d_W_;
  // Element (i,j) sits at i + j*n, so the diagonal is exactly the multiples of
  // n+1 within [0, n*n).
  const int diag_stride_;
  R cv_[NP + 1];
  R cw_[NP + 1];

  __device__ void operator()(const int idx) const {
    T v = ops::zero();
    T w = ops::zero();

#pragma unroll
    for (int k = 0; k < NP; ++k) {
      const T x = d_P_[k][idx];
      v = ops::fma_r(cv_[k + 1], x, v);
      w = ops::fma_r(cw_[k + 1], x, w);
    }

    if (idx % diag_stride_ == 0) {
      v = ops::add_r(v, cv_[0]);
      w = ops::add_r(w, cw_[0]);
    }

    d_V_[idx] = v;
    d_W_[idx] = w;
  }
};

// Build a PadeEvenOddFunctor whose const array members are initialised from the
// host coefficient arrays. The arrays are const, so they must be filled in the
// aggregate initialiser rather than assigned after the fact; a small index-pack
// helper does that for the up-to-four powers the ladder ever uses.
template<typename T, int NP, std::size_t... Ps, std::size_t... Cs>
PadeEvenOddFunctor<T, NP> make_even_odd(const T *const *d_P, int diag_stride,
                                        const typename elem_ops<T>::real_type *cv,
                                        const typename elem_ops<T>::real_type *cw,
                                        T *d_V, T *d_W, std::index_sequence<Ps...>,
                                        std::index_sequence<Cs...>) {
  using R = typename elem_ops<T>::real_type;
  return PadeEvenOddFunctor<T, NP>{
      {d_P[Ps]...}, d_V, d_W, diag_stride, {static_cast<R>(cv[Cs])...}, {static_cast<R>(cw[Cs])...}};
}

template<typename T, int NP>
void launch_even_odd(const wwr::wwrStream_t stream, const int n, const T *const *d_P,
                     const typename elem_ops<T>::real_type *cv,
                     const typename elem_ops<T>::real_type *cw, T *d_V, T *d_W) {
  const auto functor = make_even_odd<T, NP>(d_P, n + 1, cv, cw, d_V, d_W,
                                            std::make_index_sequence<NP>{},
                                            std::make_index_sequence<NP + 1>{});
  wwr::extension::parallel_for<int>(stream, n * n, functor);
}

// ── P = V + U (numerator p(A)), Q = V - U (denominator q(A) = p(-A)) ─────────

template<typename T>
struct PadeSplitFunctor {
  using ops = elem_ops<T>;

  const T *const d_U_;
  const T *const d_V_;
  T *const d_P_;
  const int ldp_;
  T *const d_Q_;
  const int n_;

  __device__ void operator()(const int idx) const {
    // Both loads happen before either store, so d_Q_ aliasing d_V_ is safe:
    // element idx is touched only by this thread.
    const T u = d_U_[idx];
    const T v = d_V_[idx];

    const int row = idx % n_;
    const int col = idx / n_;

    d_P_[row + col * ldp_] = ops::add(v, u);
    d_Q_[idx] = ops::sub(v, u);
  }
};

// ── widen gebal's real scale vector into element-typed D and D^-1 ────────────

template<typename T>
struct ExpandScaleFunctor {
  using ops = elem_ops<T>;
  using R = typename ops::real_type;

  const R *const d_scale_;
  T *const d_diag_;
  T *const d_inv_;

  __device__ void operator()(const int idx) const {
    const R d = d_scale_[idx];
    d_diag_[idx] = ops::from_r(d);
    // gebal only ever reports powers of two, so this division is exact.
    d_inv_[idx] = ops::from_r(R(1) / d);
  }
};

// ── 1-norm support: a modulus pre-transform and a plus fold for reduce_columns ─

template<typename T, typename R>
struct ModulusFunctor {
  __device__ R operator()(const T &x) const { return elem_ops<T>::modulus(x); }
};

template<typename R>
struct PlusFunctor {
  __device__ R operator()(const R a, const R b) const { return a + b; }
};

// ── max reduction over the column sums ───────────────────────────────────────
//
// One block, grid-stride load, shared-memory tree. n is bounded by kMaxDim
// (46340, enforced in the interface), so a single block always suffices and the
// whole reduction is one launch with no temporary allocation.
//
// Hand-written rather than a Thrust reduce, following the same decision
// calaman.reduce_columns and WarpWraps's parallel_for.cuh document: a Thrust
// *algorithm* broke under relocatable device code, so this codebase reduces by
// hand. (The original of this routine hit the identical failure -- a
// thrust::reduce with thrust::maximum raised cudaErrorInvalidDeviceFunction
// across the full device link -- and replaced it with exactly this kernel.)
constexpr int kMaxThreads = 256;

template<typename R>
__global__ void max_reduce_kernel(const int n, R *const d_values) {
  __shared__ R sdata[kMaxThreads];
  const int tid = static_cast<int>(threadIdx.x);

  // Every load happens before the first barrier, so writing d_values[0] at the
  // end cannot race with a read.
  R best = R(0);
  for (int i = tid; i < n; i += kMaxThreads) {
    const R v = d_values[i];
    // NaN must win and then stay won: once best is NaN, a finite v fails both
    // tests and leaves it alone. A plain v > best would silently report a finite
    // norm for a matrix holding NaN.
    if (v > best || isnan(v)) {
      best = v;
    }
  }
  sdata[tid] = best;
  __syncthreads();

  for (int s = kMaxThreads / 2; s > 0; s >>= 1) {
    if (tid < s) {
      const R other = sdata[tid + s];
      if (other > sdata[tid] || isnan(other)) {
        sdata[tid] = other;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_values[0] = sdata[0];
  }
}

} // namespace

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Launchers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

template<typename T, typename R>
void pade_even_odd(const wwr::wwrStream_t stream, const int n, const int num_powers, const T *d_P1,
                   const T *d_P2, const T *d_P3, const T *d_P4, const R *cv, const R *cw, T *d_V,
                   T *d_W) {
  if (n < 1) {
    return;
  }
  const T *d_P[4] = {d_P1, d_P2, d_P3, d_P4};
  switch (num_powers) {
  case 1:
    launch_even_odd<T, 1>(stream, n, d_P, cv, cw, d_V, d_W);
    break;
  case 2:
    launch_even_odd<T, 2>(stream, n, d_P, cv, cw, d_V, d_W);
    break;
  case 3:
    launch_even_odd<T, 3>(stream, n, d_P, cv, cw, d_V, d_W);
    break;
  case 4:
    launch_even_odd<T, 4>(stream, n, d_P, cv, cw, d_V, d_W);
    break;
  default:
    break; // pade() never asks for anything else
  }
}

template<typename T>
void pade_split(const wwr::wwrStream_t stream, const int n, const T *d_U, const T *d_V, T *d_P,
                const int ldp, T *d_Q) {
  if (n < 1) {
    return;
  }
  const PadeSplitFunctor<T> functor{d_U, d_V, d_P, ldp, d_Q, n};
  wwr::extension::parallel_for<int>(stream, n * n, functor);
}

template<typename T, typename R>
void expand_scale(const wwr::wwrStream_t stream, const int n, const R *d_scale, T *d_diag,
                  T *d_inv) {
  if (n < 1) {
    return;
  }
  const ExpandScaleFunctor<T> functor{d_scale, d_diag, d_inv};
  wwr::extension::parallel_for<int>(stream, n, functor);
}

template<typename T, typename R>
void abs_colsums(const wwr::wwrStream_t stream, const int n, const T *d_A, const int lda,
                 R *d_colsum) {
  if (n < 1) {
    return;
  }
  const std::size_t order = static_cast<std::size_t>(n);
  reduce_columns_transform<T, R>(stream, d_A, d_colsum, order, order,
                                 static_cast<std::size_t>(lda), ModulusFunctor<T, R>{},
                                 PlusFunctor<R>{});
}

template<typename R>
void max_reduce(const wwr::wwrStream_t stream, const int n, R *d_values) {
  if (n < 1) {
    return;
  }
  max_reduce_kernel<R><<<1, kMaxThreads, 0, stream>>>(n, d_values);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Explicit instantiations -- one per supported type, matching expm_bridge.h's
// declarations and the module's use sites. This list and interface.cppm's
// extern-template list must stay in step.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

template void pade_even_odd<float, float>(wwr::wwrStream_t, int, int, const float *, const float *,
                                          const float *, const float *, const float *, const float *,
                                          float *, float *);
template void pade_even_odd<double, double>(wwr::wwrStream_t, int, int, const double *,
                                            const double *, const double *, const double *,
                                            const double *, const double *, double *, double *);
template void pade_even_odd<wwrFloatComplex, float>(wwr::wwrStream_t, int, int,
                                                    const wwrFloatComplex *, const wwrFloatComplex *,
                                                    const wwrFloatComplex *, const wwrFloatComplex *,
                                                    const float *, const float *, wwrFloatComplex *,
                                                    wwrFloatComplex *);
template void pade_even_odd<wwrDoubleComplex, double>(
    wwr::wwrStream_t, int, int, const wwrDoubleComplex *, const wwrDoubleComplex *,
    const wwrDoubleComplex *, const wwrDoubleComplex *, const double *, const double *,
    wwrDoubleComplex *, wwrDoubleComplex *);

template void pade_split<float>(wwr::wwrStream_t, int, const float *, const float *, float *, int,
                                float *);
template void pade_split<double>(wwr::wwrStream_t, int, const double *, const double *, double *,
                                 int, double *);
template void pade_split<wwrFloatComplex>(wwr::wwrStream_t, int, const wwrFloatComplex *,
                                          const wwrFloatComplex *, wwrFloatComplex *, int,
                                          wwrFloatComplex *);
template void pade_split<wwrDoubleComplex>(wwr::wwrStream_t, int, const wwrDoubleComplex *,
                                           const wwrDoubleComplex *, wwrDoubleComplex *, int,
                                           wwrDoubleComplex *);

template void expand_scale<float, float>(wwr::wwrStream_t, int, const float *, float *, float *);
template void expand_scale<double, double>(wwr::wwrStream_t, int, const double *, double *,
                                           double *);
template void expand_scale<wwrFloatComplex, float>(wwr::wwrStream_t, int, const float *,
                                                   wwrFloatComplex *, wwrFloatComplex *);
template void expand_scale<wwrDoubleComplex, double>(wwr::wwrStream_t, int, const double *,
                                                     wwrDoubleComplex *, wwrDoubleComplex *);

template void abs_colsums<float, float>(wwr::wwrStream_t, int, const float *, int, float *);
template void abs_colsums<double, double>(wwr::wwrStream_t, int, const double *, int, double *);
template void abs_colsums<wwrFloatComplex, float>(wwr::wwrStream_t, int, const wwrFloatComplex *,
                                                  int, float *);
template void abs_colsums<wwrDoubleComplex, double>(wwr::wwrStream_t, int, const wwrDoubleComplex *,
                                                    int, double *);

template void max_reduce<float>(wwr::wwrStream_t, int, float *);
template void max_reduce<double>(wwr::wwrStream_t, int, double *);

} // namespace calaman::device
