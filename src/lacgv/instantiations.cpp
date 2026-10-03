/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of lacgv per supported complex type
 *
 * Implementation unit of calaman.lacgv. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the wrapper again -- its body names the device launcher declared
 * only in the interface's global module fragment, so it must be instantiated
 * here, inside this library. The device-side work it calls is instantiated
 * separately, in lacgv.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing.
 */

module calaman.lacgv;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without these.
import wwr.runtime_api;
import wwr.complex; // wwrFloatComplex, wwrDoubleComplex

namespace calaman {

template Status lacgv<wwr::wwrFloatComplex>(wwr::wwrStream_t, int, wwr::wwrFloatComplex *, int);
template Status lacgv<wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, wwr::wwrDoubleComplex *, int);

} // namespace calaman
