/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of paterson_stockmeyer() and its workspace
 *        helpers per type
 *
 * Implementation unit of calaman.paterson_stockmeyer. Pairs with the
 * `extern template` declarations in interface.cppm: together they keep every
 * importer from instantiating the wrapper again -- its body names the device
 * launcher (device::paterson_stockmeyer_build_block) declared only in the
 * interface's global module fragment, so it must be instantiated here, inside
 * this library. The device-side work it calls is instantiated separately, in
 * paterson_stockmeyer.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing.
 */

module calaman.paterson_stockmeyer;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without these.
import std;
import wwr.blas;
import wwr.runtime_api;
import wwr.wrappers.blas;
import calaman.common;

namespace calaman {

template std::size_t paterson_stockmeyer_bufferSize<float>(int, int, int);
template std::size_t paterson_stockmeyer_bufferSize<double>(int, int, int);

template std::size_t paterson_stockmeyer_power_stride<float>(int);
template std::size_t paterson_stockmeyer_power_stride<double>(int);

template wwr::wwrblasStatus_t paterson_stockmeyer<float>(wwr::wwrblasHandle_t, int, const float *,
                                                         int, const float *, int, float *, int,
                                                         void *, std::size_t, int);
template wwr::wwrblasStatus_t paterson_stockmeyer<double>(wwr::wwrblasHandle_t, int, const double *,
                                                          int, const double *, int, double *, int,
                                                          void *, std::size_t, int);

} // namespace calaman
