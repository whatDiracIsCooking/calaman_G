/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of lascl per supported type
 *
 * Implementation unit of calaman.lascl. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the wrapper again -- its body names the device launcher
 * declared only in the interface's global module fragment, so it must be
 * instantiated here, inside this library. The device-side work it calls is
 * instantiated separately, in lascl.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing.
 */

module calaman.lascl;

// An implementation unit implicitly imports its primary interface, but an
// import is not re-exported through it -- the names in the signatures below are
// not visible without these.
import std;
import wwr.runtime_api;
import wwr.complex; // wwrFloatComplex, wwrDoubleComplex

namespace calaman {

template Status lascl<float>(wwr::wwrStream_t, float, float, std::size_t, std::size_t, float *,
                             std::size_t);
template Status lascl<double>(wwr::wwrStream_t, double, double, std::size_t, std::size_t, double *,
                              std::size_t);
template Status lascl<wwr::wwrFloatComplex>(wwr::wwrStream_t, float, float, std::size_t,
                                            std::size_t, wwr::wwrFloatComplex *, std::size_t);
template Status lascl<wwr::wwrDoubleComplex>(wwr::wwrStream_t, double, double, std::size_t,
                                             std::size_t, wwr::wwrDoubleComplex *, std::size_t);

} // namespace calaman
