/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of horner()/horner_bufferSize() per type
 *
 * Implementation unit of calaman.horner. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the wrapper again -- its body names the device launchers
 * (device::horner_set_scaled_identity / _add_scaled_identity) declared only in
 * the interface's global module fragment, so it must be instantiated here,
 * inside this library. The device-side work it calls is instantiated
 * separately, in horner.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing.
 */

module calaman.horner;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without these.
import std;
import wwr.blas;
import wwr.runtime_api;
import wwr.wrappers.blas;
import calaman.common;

namespace calaman {

template std::size_t horner_bufferSize<float>(int);
template std::size_t horner_bufferSize<double>(int);

template wwr::wwrblasStatus_t horner<float>(wwr::wwrblasHandle_t, int, const float *, int,
                                            const float *, int, float *, int, void *,
                                            std::size_t);
template wwr::wwrblasStatus_t horner<double>(wwr::wwrblasHandle_t, int, const double *, int,
                                             const double *, int, double *, int, void *,
                                             std::size_t);

} // namespace calaman
