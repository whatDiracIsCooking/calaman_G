/**
 * @file interface.cppm
 * @brief Primary interface for calaman.test.elementwise_compare
 *
 * A GPU test utility: count how many elements of two device arrays differ,
 * wholly on the device, for checking a computed result against an expected one.
 * Two stages, both in elementwise_compare.cu -- a per-element mismatch flag
 * (parallel_for), then a one-block sum reduction of the flags. The caller owns
 * the two device arrays; the scratch flag buffer and the result scalar are
 * allocated here and freed on return.
 *
 * Comparison is exact inequality (a[i] != b[i]); there is no tolerance.
 *
 * Usage:
 *   import calaman.test.elementwise_compare;
 *   using namespace wwr::extension;
 *
 *   auto handle = std::make_shared<DeviceHandle>(0);
 *   // d_a, d_b: device arrays of n floats, already populated
 *   unsigned int differing = calaman::test::count_mismatches(handle, d_a, d_b, n);
 *
 * `extern template` below pairs with instantiations.cpp: the wrapper is
 * instantiated once inside this library, so an importer never re-instantiates a
 * body that names the .cu-side device launchers (declared only in the GMF).
 */

module;

#include "elementwise_compare_bridge.h"

export module calaman.test.elementwise_compare;

import std;
import wwr.runtime_api;
import wwr.extension.error_handling;
import wwr.extension.runtime;
import wwr.extension.memory_buffer;

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace.
namespace calaman::test {

/**
 * @brief Count how many of the first `n` elements of `d_a` and `d_b` differ
 *
 * Runs both device stages on the handle's stream, copies the single count back
 * and synchronizes before returning it. Returns 0 without launching anything
 * when `n` is 0.
 *
 * @tparam T Element type; one of the explicitly instantiated types (float, double)
 * @param handle Device whose stream the work runs on and whose pool the scratch
 *               buffers draw from; must be the device `d_a`/`d_b` live on
 * @param d_a First device array of at least `n` elements
 * @param d_b Second device array of at least `n` elements
 * @param n Number of elements to compare
 * @return Number of indices i in [0, n) where d_a[i] != d_b[i]
 */
export template<typename T>
unsigned int count_mismatches(std::shared_ptr<wwr::extension::DeviceHandle> handle, const T *d_a,
                              const T *d_b, const std::size_t n) {
  if (n == 0) {
    return 0u;
  }

  using DeviceAbort = wwr::extension::AbortPolicy<wwr::wwrError_t>;
  using HostAbort = wwr::extension::AbortPolicy<wwr::extension::stdHostMemoryError_t>;

  const auto stream = handle->stream().get();

  wwr::extension::DeviceBufferWrapper<unsigned int, DeviceAbort, DeviceAbort> diffs(n, handle);
  wwr::extension::DeviceBufferWrapper<unsigned int, DeviceAbort, DeviceAbort> result(1, handle);

  device::write_mismatch_flags(stream, d_a, d_b, diffs.data(), n);
  device::reduce_count(stream, diffs.data(), result.data(), n);

  wwr::extension::HostBufferWrapper<unsigned int, HostAbort, HostAbort> host_count(1);
  wwr::extension::copy(host_count, result, stream);
  wwr::wwrStreamSynchronize(stream);
  return host_count.data()[0];
}

extern template unsigned int count_mismatches<float>(std::shared_ptr<wwr::extension::DeviceHandle>,
                                                     const float *, const float *, std::size_t);
extern template unsigned int count_mismatches<double>(std::shared_ptr<wwr::extension::DeviceHandle>,
                                                      const double *, const double *, std::size_t);

/**
 * @brief The largest |d_a[i] - d_b[i]| over the first `n` elements
 *
 * The magnitude metric for a tolerance check: assert it is at most the
 * tolerance a numerical comparison allows (e.g. GPU result vs reference LAPACK).
 * Runs the abs-diff stage then a max reduction on the handle's stream; returns
 * 0 without launching anything when `n` is 0.
 *
 * @tparam T Element type; one of the explicitly instantiated types (float, double)
 * @param handle Device whose stream the work runs on and whose pool the scratch
 *               buffer draws from; must be the device d_a/d_b live on
 * @param d_a First device array of at least `n` elements
 * @param d_b Second device array of at least `n` elements
 * @param n Number of elements to compare
 * @return max over i in [0, n) of |d_a[i] - d_b[i]|
 */
export template<typename T>
T max_abs_diff(std::shared_ptr<wwr::extension::DeviceHandle> handle, const T *d_a, const T *d_b,
               const std::size_t n) {
  if (n == 0) {
    return T(0);
  }

  using DeviceAbort = wwr::extension::AbortPolicy<wwr::wwrError_t>;
  using HostAbort = wwr::extension::AbortPolicy<wwr::extension::stdHostMemoryError_t>;

  const auto stream = handle->stream().get();

  wwr::extension::DeviceBufferWrapper<T, DeviceAbort, DeviceAbort> abs_diff(n, handle);
  wwr::extension::DeviceBufferWrapper<T, DeviceAbort, DeviceAbort> result(1, handle);

  device::write_abs_diff(stream, d_a, d_b, abs_diff.data(), n);
  device::reduce_max(stream, abs_diff.data(), result.data(), n);

  wwr::extension::HostBufferWrapper<T, HostAbort, HostAbort> host_max(1);
  wwr::extension::copy(host_max, result, stream);
  wwr::wwrStreamSynchronize(stream);
  return host_max.data()[0];
}

extern template float max_abs_diff<float>(std::shared_ptr<wwr::extension::DeviceHandle>,
                                          const float *, const float *, std::size_t);
extern template double max_abs_diff<double>(std::shared_ptr<wwr::extension::DeviceHandle>,
                                            const double *, const double *, std::size_t);

/**
 * @brief How many of the first `n` elements differ by more than the tolerance
 *
 * Counts indices where |d_a[i] - d_b[i]| > atol + rtol*|d_b[i]| -- the
 * float-friendly counterpart to count_mismatches, which is exact. `d_b` is the
 * reference side the relative term scales against. With the default
 * atol = rtol = 0 it counts strict magnitude differences; note that unlike
 * count_mismatches, a NaN compared with itself does NOT count here (NaN > 0 is
 * false). Runs the tolerance stage then the sum reduction on the handle's
 * stream; returns 0 without launching anything when `n` is 0.
 *
 * @tparam T Element type; one of the explicitly instantiated types (float, double)
 * @param handle Device whose stream the work runs on and whose pool the scratch
 *               buffers draw from; must be the device d_a/d_b live on
 * @param d_a First device array of at least `n` elements
 * @param d_b Reference device array of at least `n` elements
 * @param n Number of elements to compare
 * @param atol Absolute tolerance
 * @param rtol Relative tolerance, scaling |d_b[i]|; defaults to 0
 * @return count of i in [0, n) with |d_a[i]-d_b[i]| > atol + rtol*|d_b[i]|
 */
export template<typename T>
unsigned int count_beyond_tolerance(std::shared_ptr<wwr::extension::DeviceHandle> handle,
                                    const T *d_a, const T *d_b, const std::size_t n, const T atol,
                                    const T rtol = T(0)) {
  if (n == 0) {
    return 0u;
  }

  using DeviceAbort = wwr::extension::AbortPolicy<wwr::wwrError_t>;
  using HostAbort = wwr::extension::AbortPolicy<wwr::extension::stdHostMemoryError_t>;

  const auto stream = handle->stream().get();

  wwr::extension::DeviceBufferWrapper<unsigned int, DeviceAbort, DeviceAbort> flags(n, handle);
  wwr::extension::DeviceBufferWrapper<unsigned int, DeviceAbort, DeviceAbort> result(1, handle);

  device::write_tolerance_flags(stream, d_a, d_b, flags.data(), n, atol, rtol);
  device::reduce_count(stream, flags.data(), result.data(), n);

  wwr::extension::HostBufferWrapper<unsigned int, HostAbort, HostAbort> host_count(1);
  wwr::extension::copy(host_count, result, stream);
  wwr::wwrStreamSynchronize(stream);
  return host_count.data()[0];
}

extern template unsigned int
count_beyond_tolerance<float>(std::shared_ptr<wwr::extension::DeviceHandle>, const float *,
                              const float *, std::size_t, float, float);
extern template unsigned int
count_beyond_tolerance<double>(std::shared_ptr<wwr::extension::DeviceHandle>, const double *,
                               const double *, std::size_t, double, double);

} // namespace calaman::test
