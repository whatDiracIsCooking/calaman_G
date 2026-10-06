/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of lacp2() per complex type
 *
 * Implementation unit of calaman.lacp2, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::lacp2, so it is
 * instantiated once here. Keep this list in step with lacp2.cu's.
 */

module calaman.lacp2;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;
import calaman.common; // Region

namespace calaman {

template Status lacp2<wwr::wwrFloatComplex>(wwr::wwrStream_t, Region, std::size_t, std::size_t,
                                            const float *, std::size_t, wwr::wwrFloatComplex *,
                                            std::size_t);
template Status lacp2<wwr::wwrDoubleComplex>(wwr::wwrStream_t, Region, std::size_t, std::size_t,
                                             const double *, std::size_t,
                                             wwr::wwrDoubleComplex *, std::size_t);

} // namespace calaman
