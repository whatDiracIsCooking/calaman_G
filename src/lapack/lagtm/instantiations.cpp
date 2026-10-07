/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of lagtm() per type
 *
 * Implementation unit of calaman.lagtm, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::lagtm, so it is
 * instantiated once here. Keep this list in step with lagtm.cu's.
 */

module calaman.lagtm;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex
import calaman.common; // Trans

namespace calaman {

template Status lagtm<float>(wwr::wwrStream_t, Trans, std::size_t, std::size_t, float,
                             const float *, const float *, const float *, const float *,
                             std::size_t, float, float *, std::size_t);
template Status lagtm<double>(wwr::wwrStream_t, Trans, std::size_t, std::size_t, double,
                              const double *, const double *, const double *, const double *,
                              std::size_t, double, double *, std::size_t);
template Status lagtm<wwr::wwrFloatComplex>(
    wwr::wwrStream_t, Trans, std::size_t, std::size_t, float, const wwr::wwrFloatComplex *,
    const wwr::wwrFloatComplex *, const wwr::wwrFloatComplex *, const wwr::wwrFloatComplex *,
    std::size_t, float, wwr::wwrFloatComplex *, std::size_t);
template Status lagtm<wwr::wwrDoubleComplex>(
    wwr::wwrStream_t, Trans, std::size_t, std::size_t, double, const wwr::wwrDoubleComplex *,
    const wwr::wwrDoubleComplex *, const wwr::wwrDoubleComplex *, const wwr::wwrDoubleComplex *,
    std::size_t, double, wwr::wwrDoubleComplex *, std::size_t);

} // namespace calaman
