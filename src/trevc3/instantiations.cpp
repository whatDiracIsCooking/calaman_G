/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of trevc3 per supported type
 *
 * Implementation unit of calaman.trevc3. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the wrapper again -- its body names the device launcher declared
 * only in the interface's global module fragment, so it must be instantiated
 * here, inside this library. The device-side work it calls is instantiated
 * separately, in trevc3.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing. Real only: the surface is ?strevc3 / ?dtrevc3.
 */

module calaman.trevc3;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without this.
import wwr.runtime_api;

namespace calaman {

template Status trevc3<float>(wwr::wwrStream_t, bool, bool, int, const float *, int, float *, int,
                              float *, int, float *);
template Status trevc3<double>(wwr::wwrStream_t, bool, bool, int, const double *, int, double *, int,
                               double *, int, double *);

} // namespace calaman
