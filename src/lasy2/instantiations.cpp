/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of lasy2 per supported type
 *
 * Implementation unit of calaman.lasy2. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the wrapper again -- its body names the device launcher declared
 * only in the interface's global module fragment, so it must be instantiated
 * here, inside this library. The device-side work it calls is instantiated
 * separately, in lasy2.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing. Real only: LAPACK ships no complex ?lasy2.
 */

module calaman.lasy2;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without this.
import wwr.runtime_api;

namespace calaman {

template Status lasy2<float>(wwr::wwrStream_t, bool, bool, int, int, int, const float *, int,
                             const float *, int, const float *, int, float *, float *, int, float *,
                             int *);
template Status lasy2<double>(wwr::wwrStream_t, bool, bool, int, int, int, const double *, int,
                              const double *, int, const double *, int, double *, double *, int,
                              double *, int *);

} // namespace calaman
