/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of la_wwaddw() per type
 *
 * Implementation unit of calaman.la_wwaddw, paired with interface.cppm's
 * `extern template` list: the wrapper names the GMF-declared device::la_wwaddw
 * launcher, so it is instantiated once here, inside this library. This list and
 * la_wwaddw.cu's must stay in step -- a type added here alone links against
 * nothing.
 */

module calaman.la_wwaddw;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex; // wwrFloatComplex, wwrDoubleComplex

namespace calaman {

template Status la_wwaddw<float>(wwr::wwrStream_t, std::size_t, float *, float *, const float *);
template Status la_wwaddw<double>(wwr::wwrStream_t, std::size_t, double *, double *,
                                  const double *);
template Status la_wwaddw<wwr::wwrFloatComplex>(wwr::wwrStream_t, std::size_t,
                                                wwr::wwrFloatComplex *, wwr::wwrFloatComplex *,
                                                const wwr::wwrFloatComplex *);
template Status la_wwaddw<wwr::wwrDoubleComplex>(wwr::wwrStream_t, std::size_t,
                                                 wwr::wwrDoubleComplex *, wwr::wwrDoubleComplex *,
                                                 const wwr::wwrDoubleComplex *);

} // namespace calaman
