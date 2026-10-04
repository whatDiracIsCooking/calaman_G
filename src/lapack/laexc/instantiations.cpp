/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of laexc per supported type
 *
 * Implementation unit of calaman.laexc. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the wrapper again -- its body names the device launcher declared
 * only in the interface's global module fragment, so it must be instantiated
 * here, inside this library. The device-side work it calls is instantiated
 * separately, in laexc.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing. Real only: LAPACK ships no complex ?laexc.
 */

module calaman.laexc;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without this.
import wwr.runtime_api;

namespace calaman {

template Status laexc<float>(wwr::wwrStream_t, bool, int, float *, int, float *, int, int, int, int,
                             int *);
template Status laexc<double>(wwr::wwrStream_t, bool, int, double *, int, double *, int, int, int,
                              int, int *);

} // namespace calaman
