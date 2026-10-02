/**
 * @file elem_ops.cuh
 * @brief One portable spelling of per-element scalar arithmetic over the four
 *        element types -- the seam that lets a kernel body read the same for
 *        real and complex
 *
 * `calaman::device::elem_ops<T>` is a trait struct of `__device__` static
 * functions -- zero/one/from_real, add/sub/mul/div, conj, real_part, imag_part,
 * modulus, scale, exp -- with a primary template for the real types (float,
 * double) and explicit specializations for wwrFloatComplex / wwrDoubleComplex. A
 * generic kernel writes `ops::add(a, b)` once and never sees `+` vs `wwrCadd`;
 * the trait picks the spelling off the element type. (`imag_part` is 0 for a
 * real T, so |Re| + |Im|-style code stays type-generic.)
 *
 * `make_complex(re, im)` is a COMPLEX-ONLY companion -- building an element from
 * two independent real components has no real-T meaning, so it is a free
 * overload set beside the trait, not a member that the real primary would have
 * to fake. It is the one remaining place the precision-divergent make_wwr*Complex
 * is spelled, which is why complex_cast-style component packing goes through it.
 *
 * WHY A TRAIT, NOT OPERATORS. WarpWraps settled this in complex.h's header: a
 * vendor complex is an operator-less float2 on CUDA but a class on HIP, so
 * `a * b` and brace-init are not portable, and the vendors' C-style
 * wwrCadd/wwrCmul/... are the only arithmetic that exists on both. We do not own
 * those types, so we cannot portably overload their operators -- a trait is how
 * the two vocabularies get one name.
 *
 * THE ASYMMETRY IT RECONCILES. The two families carry *different-shaped*
 * portable vocabularies, and nothing in WarpWraps bridges them:
 *   - basic arithmetic -- reals use the language operators, complex uses the
 *     wwrC* functions from complex.h;
 *   - transcendentals  -- reals use wwr::exp / wwr::cos / ... from math.cuh,
 *     which static_assert float/double and REFUSE a complex argument.
 * So a complex transcendental has no WarpWraps spelling at all: exp(z) here is
 * hand-built as exp(re)*(cos im, sin im) from the real wwr:: functions. That is
 * the whole reason this header exists on the calaman side -- math.cuh stops at
 * float/double on purpose and hands complex to the consumer.
 *
 * The real primary template forwards transcendentals to math.cuh (wwr::exp,
 * wwr::fabs) rather than branching ::expf vs ::exp itself -- that branch already
 * lives there, keyed on T. The two complex specializations differ only by type
 * and the vendor functions' f-suffix, so one token-pasting macro stamps both:
 * float and double cannot drift, which hand-copied specializations (the old
 * per-module elem_ops in expm.cu / gebal.cu / complex_cast.cu) could.
 *
 * SCOPE: the CORE every kernel wants. A module-specific op -- expm's real-scaled
 * fma, gebal's abs1 / is_zero -- stays local to that .cu as a free `__device__`
 * helper over this core (see expm.cu's fma_real / add_real), not a member here;
 * a kitchen-sink trait would make every consumer instantiate ops it never calls.
 *
 * DEVICE ONLY. Every member is `__device__`, and the complex builders/arithmetic
 * exist only in a device pass (complex.h gates them behind __CUDACC__/__HIP__),
 * so this is a .cuh for a .cu, never imported into a host module purview -- the
 * same shape as reduce_columns.cuh. Reached root-relative as
 * "elem_ops/elem_ops.cuh"; link calaman.elem_ops for the include root and
 * wwr.device (which carries complex.h and math.cuh).
 */

#pragma once

#include "complex.h"              // wwr complex types + wwrC* arithmetic (device pass)
#include "wrappers/math/math.cuh" // wwr::exp / cos / sin / fabs (device, float/double)

#include <type_traits>

namespace calaman::device {

/// @brief Per-element arithmetic for a real element type (float or double)
///
/// The ops are the native operators and math.cuh's real transcendentals;
/// real_part, conj and from_real are identities, modulus is the absolute value.
/// Any T that is not float, double, or a wwr complex type lands here and is
/// rejected by the static_assert rather than silently compiling nonsense.
template<typename T>
struct elem_ops {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "calaman::device::elem_ops supports float, double, wwrFloatComplex, "
                "wwrDoubleComplex only");

  using real_type = T;

  static __device__ __forceinline__ T zero() { return T(0); }
  static __device__ __forceinline__ T one() { return T(1); }
  static __device__ __forceinline__ T from_real(const T r) { return r; }

  static __device__ __forceinline__ T add(const T a, const T b) { return a + b; }
  static __device__ __forceinline__ T sub(const T a, const T b) { return a - b; }
  static __device__ __forceinline__ T mul(const T a, const T b) { return a * b; }
  static __device__ __forceinline__ T div(const T a, const T b) { return a / b; }

  static __device__ __forceinline__ T conj(const T a) { return a; }
  static __device__ __forceinline__ T scale(const T a, const T s) { return a * s; }
  static __device__ __forceinline__ T real_part(const T a) { return a; }
  static __device__ __forceinline__ T imag_part(const T) { return T(0); }
  static __device__ __forceinline__ T modulus(const T a) { return wwr::fabs(a); }

  static __device__ __forceinline__ T exp(const T a) { return wwr::exp(a); }
};

// The two complex specializations differ only by the element type (CT), its real
// component type (RT), and the f-suffix the vendors put on the single-precision
// functions (SUF -- `f` for float, empty for double). One macro stamps both so
// the precisions cannot drift; #undef'd below so it does not leak. make_##CT is
// the builder (make_wwrFloatComplex / make_wwrDoubleComplex); wwrCreal##SUF etc.
// are complex.h's accessors and arithmetic; exp is hand-built from math.cuh's
// REAL exp/cos/sin, since math.cuh refuses a complex argument.
#define CLM_DEFINE_COMPLEX_ELEM_OPS(CT, RT, SUF)                                                    \
  template<>                                                                                        \
  struct elem_ops<wwr::CT> {                                                                        \
    using real_type = RT;                                                                           \
    static __device__ __forceinline__ wwr::CT zero() { return wwr::make_##CT(RT(0), RT(0)); }       \
    static __device__ __forceinline__ wwr::CT one() { return wwr::make_##CT(RT(1), RT(0)); }        \
    static __device__ __forceinline__ wwr::CT from_real(const RT r) {                               \
      return wwr::make_##CT(r, RT(0));                                                               \
    }                                                                                               \
    static __device__ __forceinline__ wwr::CT add(const wwr::CT a, const wwr::CT b) {               \
      return wwr::wwrCadd##SUF(a, b);                                                                \
    }                                                                                               \
    static __device__ __forceinline__ wwr::CT sub(const wwr::CT a, const wwr::CT b) {               \
      return wwr::wwrCsub##SUF(a, b);                                                                \
    }                                                                                               \
    static __device__ __forceinline__ wwr::CT mul(const wwr::CT a, const wwr::CT b) {               \
      return wwr::wwrCmul##SUF(a, b);                                                                \
    }                                                                                               \
    static __device__ __forceinline__ wwr::CT div(const wwr::CT a, const wwr::CT b) {               \
      return wwr::wwrCdiv##SUF(a, b);                                                                \
    }                                                                                               \
    static __device__ __forceinline__ wwr::CT conj(const wwr::CT a) { return wwr::wwrConj##SUF(a); } \
    static __device__ __forceinline__ wwr::CT scale(const wwr::CT a, const RT s) {                  \
      return wwr::make_##CT(wwr::wwrCreal##SUF(a) * s, wwr::wwrCimag##SUF(a) * s);                   \
    }                                                                                               \
    static __device__ __forceinline__ RT real_part(const wwr::CT a) {                               \
      return wwr::wwrCreal##SUF(a);                                                                  \
    }                                                                                               \
    static __device__ __forceinline__ RT imag_part(const wwr::CT a) {                               \
      return wwr::wwrCimag##SUF(a);                                                                  \
    }                                                                                               \
    static __device__ __forceinline__ RT modulus(const wwr::CT a) { return wwr::wwrCabs##SUF(a); }  \
    static __device__ __forceinline__ wwr::CT exp(const wwr::CT a) {                                 \
      const RT er = wwr::exp(wwr::wwrCreal##SUF(a));                                                 \
      const RT theta = wwr::wwrCimag##SUF(a);                                                        \
      return wwr::make_##CT(er * wwr::cos(theta), er * wwr::sin(theta));                             \
    }                                                                                               \
  };

CLM_DEFINE_COMPLEX_ELEM_OPS(wwrFloatComplex, float, f)
CLM_DEFINE_COMPLEX_ELEM_OPS(wwrDoubleComplex, double, )

#undef CLM_DEFINE_COMPLEX_ELEM_OPS

/// @brief Build a complex value from independent real and imaginary components.
///
/// Complex-only, by overload on the component type: a "complex from two
/// components" has no real-T counterpart, so there is deliberately no real
/// overload and `make_complex(float_re, double_im)` or a scalar-T call does not
/// compile. This is the last place the precision-divergent make_wwr*Complex is
/// selected; component packing (complex_cast) and any kernel assembling an
/// element from two real planes goes through it rather than respelling it.
__device__ __forceinline__ wwr::wwrFloatComplex make_complex(const float re, const float im) {
  return wwr::make_wwrFloatComplex(re, im);
}

__device__ __forceinline__ wwr::wwrDoubleComplex make_complex(const double re, const double im) {
  return wwr::make_wwrDoubleComplex(re, im);
}

} // namespace calaman::device
