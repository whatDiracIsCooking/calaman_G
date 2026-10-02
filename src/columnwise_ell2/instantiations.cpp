/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of columnwise_ell2() per type
 *
 * Implementation unit of calaman.columnwise_ell2. Pairs with the `extern
 * template` declarations in interface.cppm: together they keep every importer
 * from instantiating the wrapper again -- its body names the device launcher
 * (device::columnwise_ell2) declared only in the interface's global module
 * fragment, so it must be instantiated here, inside this library. The
 * device-side work it calls is instantiated separately, in columnwise_ell2.cu,
 * as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing.
 */

module calaman.columnwise_ell2;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without these.
import std;
import wwr.runtime_api;

namespace calaman {

template void columnwise_ell2<float>(wwr::wwrStream_t, std::size_t, std::size_t, const float *,
                                     std::size_t, float *);
template void columnwise_ell2<double>(wwr::wwrStream_t, std::size_t, std::size_t, const double *,
                                      std::size_t, double *);

} // namespace calaman
