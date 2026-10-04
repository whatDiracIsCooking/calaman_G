/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of the ladiv driver per supported type
 *
 * Implementation unit of calaman.ladiv. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the driver again -- its body names the device launcher declared
 * only in the interface's global module fragment, so it must be instantiated
 * here, inside this library. The device-side work it calls is instantiated
 * separately, in ladiv.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing.
 */

module calaman.ladiv;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without this.
import wwr.runtime_api;

namespace calaman {

template Status ladiv<float>(wwr::wwrStream_t, std::size_t, const float *, const float *,
                             const float *, const float *, float *, float *);
template Status ladiv<double>(wwr::wwrStream_t, std::size_t, const double *, const double *,
                              const double *, const double *, double *, double *);

} // namespace calaman
