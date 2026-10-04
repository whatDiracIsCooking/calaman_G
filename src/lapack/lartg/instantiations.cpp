/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of lartg per supported type
 *
 * Implementation unit of calaman.lartg. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the wrapper again -- its body names the device launcher declared
 * only in the interface's global module fragment, so it must be instantiated
 * here, inside this library. The device-side work it calls is instantiated
 * separately, in lartg.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing.
 */

module calaman.lartg;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without this.
import wwr.runtime_api;

namespace calaman {

template Status lartg<float>(wwr::wwrStream_t, std::size_t, const float *, const float *, float *,
                             float *, float *);
template Status lartg<double>(wwr::wwrStream_t, std::size_t, const double *, const double *,
                              double *, double *, double *);

} // namespace calaman
