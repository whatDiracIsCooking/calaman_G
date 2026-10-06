/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of lag2() per (From, To) pair
 *
 * Implementation unit of calaman.lag2, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::lag2, so it is
 * instantiated once here. Keep this list in step with lag2.cu's.
 */

module calaman.lag2;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;

namespace calaman {

template Status lag2<double, float>(wwr::wwrStream_t, std::size_t, std::size_t, const double *,
                                    std::size_t, float *, std::size_t, int *);
template Status lag2<float, double>(wwr::wwrStream_t, std::size_t, std::size_t, const float *,
                                    std::size_t, double *, std::size_t, int *);
template Status lag2<wwr::wwrDoubleComplex, wwr::wwrFloatComplex>(
    wwr::wwrStream_t, std::size_t, std::size_t, const wwr::wwrDoubleComplex *, std::size_t,
    wwr::wwrFloatComplex *, std::size_t, int *);
template Status lag2<wwr::wwrFloatComplex, wwr::wwrDoubleComplex>(
    wwr::wwrStream_t, std::size_t, std::size_t, const wwr::wwrFloatComplex *, std::size_t,
    wwr::wwrDoubleComplex *, std::size_t, int *);

} // namespace calaman
