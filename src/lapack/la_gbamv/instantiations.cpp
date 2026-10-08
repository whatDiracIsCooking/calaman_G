/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of la_gbamv() per type
 *
 * Implementation unit of calaman.la_gbamv, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::la_gbamv, so it is
 * instantiated once here. Keep this list in step with la_gbamv.cu's.
 */

module calaman.la_gbamv;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex
import calaman.common; // Trans

namespace calaman {

template Status la_gbamv<float>(wwr::wwrStream_t, Trans, std::size_t, std::size_t, std::size_t,
                                std::size_t, float, const float *, std::size_t, const float *, int,
                                float, float *, int);
template Status la_gbamv<double>(wwr::wwrStream_t, Trans, std::size_t, std::size_t, std::size_t,
                                 std::size_t, double, const double *, std::size_t, const double *,
                                 int, double, double *, int);
template Status la_gbamv<wwr::wwrFloatComplex>(wwr::wwrStream_t, Trans, std::size_t, std::size_t,
                                               std::size_t, std::size_t, float,
                                               const wwr::wwrFloatComplex *, std::size_t,
                                               const wwr::wwrFloatComplex *, int, float, float *,
                                               int);
template Status la_gbamv<wwr::wwrDoubleComplex>(wwr::wwrStream_t, Trans, std::size_t, std::size_t,
                                                std::size_t, std::size_t, double,
                                                const wwr::wwrDoubleComplex *, std::size_t,
                                                const wwr::wwrDoubleComplex *, int, double,
                                                double *, int);

} // namespace calaman
