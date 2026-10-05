/**
 * @file device_functor.h
 * @brief The trivially-copyable functor concept device launchers constrain
 *        their fold and transform ops with, plus the identity pre-transform
 *
 * device_functor names the host-checkable half of a device functor's
 * contract: a trivially-copyable class type. The callability probe
 * (`{ f(a, b) } -> ValT`) is deliberately NOT here -- in host context it
 * would reject a `__device__`-only operator(), so each kernel template
 * carries its own `requires`, a device entity that accepts it. The same name
 * and split as wwr.extension.parallel_for's device_functor, minus its
 * !is_copy_assignable conjunct: these functors are plain kernel parameters,
 * never WWR_GRID_CONSTANT, so a const member is not required.
 *
 * The concept is a plain trait, so this header #includes anywhere -- a host
 * module's GMF, a host TU, or a device .cu. identity_functor is
 * `__host__ __device__` and sits behind the compiler's device-pass macros,
 * ABSENT in a host compile rather than an error -- the shape of WarpWraps'
 * complex.h, whose types are always visible and whose device wrappers are
 * gated. Consumers reach it root-relative as "common/device_functor.h";
 * there is no module face until a host interface needs one.
 */
#pragma once

#include <type_traits>

namespace calaman::device {

/// @brief A trivially-copyable class functor passable by value to a kernel
///
/// The callability and signature checks (`{ f(a, b) } -> ValT`) live on the
/// consuming kernel or device function (see the file header); the consumer's
/// @tparam docs state the shape each op must have.
template<typename F>
concept device_functor = std::is_trivially_copyable_v<F> && std::is_class_v<F>;

} // namespace calaman::device

// identity_functor only in a device-compile pass, where __host__ __device__ is
// a keyword -- absent in a host compile rather than #error, complex.h's gate.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)

namespace calaman::device {

/// @brief Pre-transform that passes each element through unchanged (ValT == T)
///
/// Seeds the identity-pre case of a transform-reduce (reduce_columns() wraps
/// it). `__host__ __device__` so the same type is usable either side of the
/// host/device line in a device TU.
template<typename T>
struct identity_functor {
  __host__ __device__ T operator()(const T &x) const { return x; }
};

} // namespace calaman::device

#endif // device-compile pass
