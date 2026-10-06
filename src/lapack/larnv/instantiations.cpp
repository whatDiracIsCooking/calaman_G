/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of larnv() per type
 *
 * Implementation unit of calaman.larnv, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::larnv, so it is
 * instantiated once here. Keep this list in step with larnv.cu's.
 */

module calaman.larnv;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;

namespace calaman {

template Status larnv<float>(wwr::wwrStream_t, int, int *, std::size_t, float *);
template Status larnv<double>(wwr::wwrStream_t, int, int *, std::size_t, double *);
template Status larnv<wwr::wwrFloatComplex>(wwr::wwrStream_t, int, int *, std::size_t,
                                            wwr::wwrFloatComplex *);
template Status larnv<wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, int *, std::size_t,
                                             wwr::wwrDoubleComplex *);

} // namespace calaman
