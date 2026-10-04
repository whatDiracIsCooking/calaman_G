/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of laln2 per supported type
 *
 * Implementation unit of calaman.laln2. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the wrapper again -- its body names the device launcher declared
 * only in the interface's global module fragment, so it must be instantiated
 * here, inside this library. The device-side work it calls is instantiated
 * separately, in laln2.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing. Real element type: the reference is
 * SLALN2 / DLALN2.
 */

module calaman.laln2;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without this.
import wwr.runtime_api;

namespace calaman {

template Status laln2<float>(wwr::wwrStream_t, bool, int, int, float, float, const float *, int,
                             float, float, const float *, int, float, float, float *, int, float *,
                             float *, int *);
template Status laln2<double>(wwr::wwrStream_t, bool, int, int, double, double, const double *, int,
                              double, double, const double *, int, double, double, double *, int,
                              double *, double *, int *);

} // namespace calaman
