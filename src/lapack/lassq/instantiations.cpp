/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of lassq per supported type
 *
 * Implementation unit of calaman.lassq. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the wrapper again -- its body names the device launcher
 * declared only in the interface's global module fragment, so it must be
 * instantiated here, inside this library. The device-side work it calls is
 * instantiated separately, in lassq.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing.
 */

module calaman.lassq;

// An implementation unit implicitly imports its primary interface, but an
// import is not re-exported through it -- the names in the signatures below are
// not visible without these.
import std;
import wwr.runtime_api;
import wwr.complex;

namespace calaman {

template Status lassq<float>(wwr::wwrStream_t, int, const float *, int, float *, float *);
template Status lassq<double>(wwr::wwrStream_t, int, const double *, int, double *, double *);
template Status lassq<wwr::wwrFloatComplex>(wwr::wwrStream_t, int, const wwr::wwrFloatComplex *,
                                            int, float *, float *);
template Status lassq<wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, const wwr::wwrDoubleComplex *,
                                             int, double *, double *);

} // namespace calaman
