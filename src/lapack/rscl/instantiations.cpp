/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of rscl per supported type
 *
 * Implementation unit of calaman.rscl. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the wrapper again -- its body names the device launcher
 * declared only in the interface's global module fragment, so it must be
 * instantiated here, inside this library. The device-side work it calls is
 * instantiated separately, in rscl.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing.
 */

module calaman.rscl;

// An implementation unit implicitly imports its primary interface, but an
// import is not re-exported through it -- the names in the signatures below are
// not visible without these.
import std;
import wwr.runtime_api;
import wwr.complex;

namespace calaman {

template Status rscl<float>(wwr::wwrStream_t, int, float, float *, int);
template Status rscl<double>(wwr::wwrStream_t, int, double, double *, int);
template Status rscl<wwr::wwrFloatComplex>(wwr::wwrStream_t, int, float, wwr::wwrFloatComplex *,
                                           int);
template Status rscl<wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, double, wwr::wwrDoubleComplex *,
                                            int);

} // namespace calaman
