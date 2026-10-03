/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of lasrt per supported type
 *
 * Implementation unit of calaman.lasrt. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the wrapper again -- its body names the device launcher declared
 * only in the interface's global module fragment, so it must be instantiated
 * here, inside this library. The device-side work it calls is instantiated
 * separately, in lasrt.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing. Real only: LAPACK ships no complex ?lasrt.
 */

module calaman.lasrt;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without these. (SortDir is re-exported by the interface, so the
// implicit import would carry it; it is named here anyway for the same
// signatures-name-their-modules reason.)
import std;
import wwr.runtime_api;
import calaman.common; // SortDir

namespace calaman {

template Status lasrt<float>(wwr::wwrStream_t, SortDir, int, float *);
template Status lasrt<double>(wwr::wwrStream_t, SortDir, int, double *);

} // namespace calaman
