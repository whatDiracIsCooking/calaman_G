/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of lahqr per supported type
 *
 * Implementation unit of calaman.lahqr. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the wrapper again -- its body names the device launcher declared
 * only in the interface's global module fragment, so it must be instantiated
 * here, inside this library. The device-side work it calls is instantiated
 * separately, in lahqr.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing. Real only: LAPACK's double-shift QR is the
 * real path.
 */

module calaman.lahqr;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without this.
import wwr.runtime_api;

namespace calaman {

template Status lahqr<float>(wwr::wwrStream_t, bool, bool, int, int, int, float *, int, float *,
                             float *, int, int, float *, int, int *);
template Status lahqr<double>(wwr::wwrStream_t, bool, bool, int, int, int, double *, int, double *,
                              double *, int, int, double *, int, int *);

} // namespace calaman
