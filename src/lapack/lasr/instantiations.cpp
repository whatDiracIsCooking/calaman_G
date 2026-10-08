/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of lasr() per type
 *
 * Implementation unit of calaman.lasr, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::lasr, so it is
 * instantiated once here. Keep this list in step with lasr.cu's.
 */

module calaman.lasr;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;
import calaman.common; // Side, Pivot, Direct

namespace calaman {

template Status lasr<float>(wwr::wwrStream_t, Side, Pivot, Direct, std::size_t, std::size_t,
                            const float *, const float *, float *, std::size_t);
template Status lasr<double>(wwr::wwrStream_t, Side, Pivot, Direct, std::size_t, std::size_t,
                             const double *, const double *, double *, std::size_t);
template Status lasr<wwr::wwrFloatComplex>(wwr::wwrStream_t, Side, Pivot, Direct, std::size_t,
                                           std::size_t, const float *, const float *,
                                           wwr::wwrFloatComplex *, std::size_t);
template Status lasr<wwr::wwrDoubleComplex>(wwr::wwrStream_t, Side, Pivot, Direct, std::size_t,
                                            std::size_t, const double *, const double *,
                                            wwr::wwrDoubleComplex *, std::size_t);

} // namespace calaman
