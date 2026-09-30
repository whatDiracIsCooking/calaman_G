/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of count_mismatches per supported type
 *
 * Implementation unit of calaman.test.elementwise_compare. Pairs with the
 * `extern template` declarations in interface.cppm: together they keep every
 * importer from instantiating the wrapper again -- its body names the device
 * launchers declared only in the interface's global module fragment, so it must
 * be instantiated here, inside this library. The device-side work it calls is
 * instantiated separately, in elementwise_compare.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing.
 */

module calaman.test.elementwise_compare;

// An implementation unit implicitly imports its primary interface, but an
// import is not re-exported through it -- the names in the signatures and the
// wrapper body below are not visible without these.
import std;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;

namespace calaman::test {

template unsigned int count_mismatches<float>(std::shared_ptr<DeviceHandle>, const float *,
                                              const float *, std::size_t);
template unsigned int count_mismatches<double>(std::shared_ptr<DeviceHandle>, const double *,
                                               const double *, std::size_t);

template float max_abs_diff<float>(std::shared_ptr<DeviceHandle>, const float *, const float *,
                                   std::size_t);
template double max_abs_diff<double>(std::shared_ptr<DeviceHandle>, const double *, const double *,
                                     std::size_t);

template unsigned int count_beyond_tolerance<float>(std::shared_ptr<DeviceHandle>, const float *,
                                                    const float *, std::size_t, float, float);
template unsigned int count_beyond_tolerance<double>(std::shared_ptr<DeviceHandle>, const double *,
                                                     const double *, std::size_t, double, double);

} // namespace calaman::test
