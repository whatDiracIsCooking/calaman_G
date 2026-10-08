/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of ilalc() per type
 *
 * Implementation unit of calaman.ilalc, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-declared device::ilalc launcher, so
 * it is instantiated once here, inside this library. This list and ilalc.cu's
 * must stay in step -- a type added here alone links against nothing.
 */

module calaman.ilalc;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex; // wwrFloatComplex, wwrDoubleComplex

namespace calaman {

template Status ilalc<float>(wwr::wwrStream_t, int, int, const float *, int, int *, void *,
                             std::size_t);
template Status ilalc<double>(wwr::wwrStream_t, int, int, const double *, int, int *, void *,
                              std::size_t);
template Status ilalc<wwr::wwrFloatComplex>(wwr::wwrStream_t, int, int,
                                            const wwr::wwrFloatComplex *, int, int *, void *,
                                            std::size_t);
template Status ilalc<wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, int,
                                             const wwr::wwrDoubleComplex *, int, int *, void *,
                                             std::size_t);

} // namespace calaman
