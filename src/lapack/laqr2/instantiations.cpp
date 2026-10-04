/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of laqr2 per supported type
 *
 * Implementation unit of calaman.laqr2. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the wrapper again -- its body names the device launcher declared
 * only in the interface's global module fragment, so it must be instantiated
 * here, inside this library. The device-side work it calls is instantiated
 * separately, in laqr2.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing. Real only: the surface is ?slaqr2 / ?dlaqr2.
 */

module calaman.laqr2;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without this.
import wwr.runtime_api;

namespace calaman {

template Status laqr2<float>(wwr::wwrStream_t, bool, bool, int, int, int, int, float *, int, int,
                             int, float *, int, int *, int *, float *, float *, float *, int, int,
                             float *, int, int, float *, int);
template Status laqr2<double>(wwr::wwrStream_t, bool, bool, int, int, int, int, double *, int, int,
                              int, double *, int, int *, int *, double *, double *, double *, int,
                              int, double *, int, int, double *, int);

} // namespace calaman
