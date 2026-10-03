/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of lange() per type
 *
 * Implementation unit of calaman.lange. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the wrapper again -- its body names the device launcher
 * (device::lange) declared only in the interface's global module fragment, so it
 * must be instantiated here, inside this library. The device-side work it calls
 * is instantiated separately, in lange.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing.
 */

module calaman.lange;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without these.
import std;
import wwr.runtime_api;
import calaman.common; // MatrixNorm

namespace calaman {

template Status lange<float>(wwr::wwrStream_t, MatrixNorm, std::size_t, std::size_t, const float *,
                             std::size_t, float *);
template Status lange<double>(wwr::wwrStream_t, MatrixNorm, std::size_t, std::size_t,
                              const double *, std::size_t, double *);

} // namespace calaman
