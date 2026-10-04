// larfx.cu
//
// The device-kernel half of calaman.larfx: the inline small-order application
// of the elementary reflector H = I - tau*v*v^T. LAPACK's ?larfx hand-unrolls
// ten bodies (one per order 1..10) so the reflector stays in registers; the GPU
// does not need the unrolling -- one thread per independent vector already keeps
// each short reduction in registers -- so the ten bodies collapse to a single
// per-vector functor. Launched through wwr.extension.parallel_for (a device-code
// header #included here, not a module), like laqp2's downdate. Shared unchanged
// between both backends.
//
// Each thread owns one vector x of length `order` (a column of C for side L, a
// row for side R), passed as a {within, between} stride pair so one functor
// serves both sides: thread t's vector starts at C + t*between and steps by
// `within`. It forms sum = sum_r v[r]*x[r], then x[r] -= tau*sum*v[r] -- the
// exact math of ?larfx's inline code (c(k,j) -= sum * (tau*v(k))). `order` is a
// runtime value <= 10, so the loops are short but not statically unrolled; that
// is a micro-optimisation ?larfx wanted for the CPU, not a correctness property.
#include "larfx_bridge.h"

#include "extension/parallel_for/parallel_for.cuh"

#include <cstddef>

namespace calaman::device {

namespace {

/// @brief Per-vector reflector-application functor invoked by parallel_for
///
/// Trivially copyable with only const scalar/pointer members, as device_functor
/// requires. operator() is const-qualified and __device__; it writes only into
/// C_ (a const pointer to mutable storage), never its own members. The vector t
/// handled by index t begins at C_ + t * between_ and its r-th entry is at
/// r * within_, so one functor drives both sides via the stride pair.
template<typename T>
struct ApplyFunctor {
  const T *const v_;             ///< reflector, length order_, stride 1
  T *const C_;                   ///< matrix base; vector t at C_ + t*between_
  const std::size_t order_;      ///< length of v and of each vector (1..10)
  const std::size_t within_;     ///< element stride within one vector
  const std::size_t between_;    ///< stride between consecutive vectors
  const T tau_;                  ///< reflector scalar (nonzero)

  __device__ void operator()(const std::size_t t) const {
    T *const x = C_ + t * between_;

    // sum = v^T x, accumulated left-to-right as ?larfx's inline code does.
    T sum = T{0};
    for (std::size_t r = 0; r < order_; ++r) {
      sum += v_[r] * x[r * within_];
    }

    // x -= (tau*sum) * v. s folds tau and the dot once, matching ?larfx's
    // c(k,j) -= sum * t(k) with t(k) = tau*v(k).
    const T s = tau_ * sum;
    for (std::size_t r = 0; r < order_; ++r) {
      x[r * within_] -= s * v_[r];
    }
  }
};

} // namespace

template<typename T>
void larfx(const wwr::wwrStream_t stream, const std::size_t count, const std::size_t order,
           const std::size_t stride_within, const std::size_t stride_between, const T *const v,
           const T tau, T *const C) {
  if (count < 1) {
    return;
  }
  const ApplyFunctor<T> functor{v, C, order, stride_within, stride_between, tau};
  wwr::extension::parallel_for<std::size_t>(stream, count, functor);
}

// One per supported type, matching larfx_bridge.h's declaration and the module's
// use sites -- float and double, as the reflector family is templated over.
template void larfx<float>(wwr::wwrStream_t, std::size_t, std::size_t, std::size_t, std::size_t,
                           const float *, float, float *);
template void larfx<double>(wwr::wwrStream_t, std::size_t, std::size_t, std::size_t, std::size_t,
                            const double *, double, double *);

} // namespace calaman::device
