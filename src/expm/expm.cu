// expm.cu
//
// The device-kernel half of calaman.expm: the genuinely per-element pieces of
// the matrix exponential -- the four the scaling-and-squaring driver needs, plus
// the eigenvalue-exponential column scale the self-adjoint path (expm_herm)
// needs. Everything else in expm is a host composition of wrapped BLAS
// (gemm/geam), the wrapped LU solve (getrf/getrs) and the wrapped symmetric /
// Hermitian eigensolver (syevd/heevd); these are the kernels those calls cannot
// express. Shared unchanged between both backends, like lacpy.cu / gebal.cu: a
// .cu is compiled by the backend compiler, so parallel_for's launch machinery,
// calaman.reduce_columns's segmented reduce, the raw <<<>>> launch syntax and
// the complex types/accessors (through complex.h) are all available directly.
//
// NEUTRAL COMPLEX, NEVER .x/.y. Per-element scalar arithmetic goes through
// calaman::device::elem_ops<T> (common/elem_ops.cuh): the primary template is
// the native operators for real T, the complex specializations route through
// complex.h's wwrCreal*/wwrCimag* accessors, wwrCabs* modulus and make_wwr*Complex
// builder. This .cu used to carry its own three elem_ops specializations (as
// gebal.cu and complex_cast.cu still do); they are now the one shared header.
//
// The Pade coefficients are REAL even when the matrix is complex, so the fused
// kernels need two mixed ops the core does not carry: "accumulate a real multiple
// of an element" and "add a real to the diagonal". Both layer on as free helpers
// over elem_ops (fma_real / add_real below) rather than bloating the shared trait.
#include "expm_bridge.h"

#include <complex.h>
#include "common/constants.h"
#include "common/elem_ops.cuh"
#include <extension/parallel_for/parallel_for.cuh>
#include "reduce_columns/reduce_columns.cuh"

#include <cmath>
#include <cstddef>
#include <utility>

namespace calaman::device {

// complex.h puts the neutral complex types in namespace wwr; pull the two type
// names in so the explicit instantiations below can spell them bare. The
// accessor / constructor calls stay wwr::-qualified.
using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

namespace {

// ── mixed real/element ops the Pade kernels need, over the shared elem_ops ────
//
// The coefficients are real, so these are the two operations that fall outside
// the type-generic core: a real-coefficient fused multiply-add and adding a real
// to the diagonal. They are expm-specific, so they live here as free helpers
// rather than members of elem_ops; each is one line over the core.

/// acc + c*x, with c real and x, acc of element type T.
template<typename T, typename R>
__device__ __forceinline__ T fma_real(const R c, const T x, const T acc) {
  return elem_ops<T>::add(acc, elem_ops<T>::scale(x, c));
}

/// a + c, adding the real c to a's real component only.
template<typename T, typename R>
__device__ __forceinline__ T add_real(const T a, const R c) {
  return elem_ops<T>::add(a, elem_ops<T>::from_real(c));
}

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
    T v = calaman::kZero<T>;
    T w = calaman::kZero<T>;

#pragma unroll
    for (int k = 0; k < NP; ++k) {
      const T x = d_P_[k][idx];
      v = fma_real(cv_[k + 1], x, v);
      w = fma_real(cw_[k + 1], x, w);
    }

    if (idx % diag_stride_ == 0) {
      v = add_real(v, cv_[0]);
      w = add_real(w, cw_[0]);
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

// ── M = U * diag(exp(w)): the middle step of the self-adjoint exponential ─────
//
// Column j of the eigenvectors U is scaled by the real scalar exp(w[j]). The
// eigenvalues w are real (A is symmetric/Hermitian), so this is a real scale of
// an element even for complex T -- elem_ops::scale(element, real). The exp is
// taken through elem_ops<R>::exp, the same real exp the core carries.
template<typename T, typename R>
struct HermExpScaleFunctor {
  using ops = elem_ops<T>;

  const T *const d_U_;
  const R *const d_w_;
  T *const d_M_;
  const int n_;

  __device__ void operator()(const int idx) const {
    const int col = idx / n_;
    const R e = elem_ops<R>::exp(d_w_[col]);
    d_M_[idx] = ops::scale(d_U_[idx], e);
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
void herm_exp_scale(const wwr::wwrStream_t stream, const int n, const T *d_U, const R *d_w,
                    T *d_M) {
  if (n < 1) {
    return;
  }
  const HermExpScaleFunctor<T, R> functor{d_U, d_w, d_M, n};
  wwr::extension::parallel_for<int>(stream, n * n, functor);
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

template void herm_exp_scale<float, float>(wwr::wwrStream_t, int, const float *, const float *,
                                           float *);
template void herm_exp_scale<double, double>(wwr::wwrStream_t, int, const double *, const double *,
                                             double *);
template void herm_exp_scale<wwrFloatComplex, float>(wwr::wwrStream_t, int, const wwrFloatComplex *,
                                                     const float *, wwrFloatComplex *);
template void herm_exp_scale<wwrDoubleComplex, double>(wwr::wwrStream_t, int,
                                                       const wwrDoubleComplex *, const double *,
                                                       wwrDoubleComplex *);

template void abs_colsums<float, float>(wwr::wwrStream_t, int, const float *, int, float *);
template void abs_colsums<double, double>(wwr::wwrStream_t, int, const double *, int, double *);
template void abs_colsums<wwrFloatComplex, float>(wwr::wwrStream_t, int, const wwrFloatComplex *,
                                                  int, float *);
template void abs_colsums<wwrDoubleComplex, double>(wwr::wwrStream_t, int, const wwrDoubleComplex *,
                                                    int, double *);

template void max_reduce<float>(wwr::wwrStream_t, int, float *);
template void max_reduce<double>(wwr::wwrStream_t, int, double *);

} // namespace calaman::device
