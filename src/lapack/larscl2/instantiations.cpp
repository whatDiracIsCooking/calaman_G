/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of larscl2() per type
 *
 * Implementation unit of calaman.larscl2, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-declared device::larscl2 launcher,
 * so it is instantiated once here, inside this library. This list and
 * larscl2.cu's must stay in step -- a type added here alone links against
 * nothing.
 */

module calaman.larscl2;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex; // wwrFloatComplex, wwrDoubleComplex

namespace calaman {

template Status larscl2<float>(wwr::wwrStream_t, std::size_t, std::size_t, const float *, float *,
                               std::size_t);
template Status larscl2<double>(wwr::wwrStream_t, std::size_t, std::size_t, const double *,
                                double *, std::size_t);
template Status larscl2<wwr::wwrFloatComplex>(wwr::wwrStream_t, std::size_t, std::size_t,
                                              const float *, wwr::wwrFloatComplex *, std::size_t);
template Status larscl2<wwr::wwrDoubleComplex>(wwr::wwrStream_t, std::size_t, std::size_t,
                                               const double *, wwr::wwrDoubleComplex *,
                                               std::size_t);

} // namespace calaman
