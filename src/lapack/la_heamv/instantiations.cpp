/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of la_heamv() per type
 *
 * Implementation unit of calaman.la_heamv, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::la_syamv, so it is
 * instantiated once here. Every type must also be in la_syamv.cu's list.
 */

module calaman.la_heamv;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex
import calaman.common; // Uplo

namespace calaman {

template Status la_heamv<wwr::wwrFloatComplex>(wwr::wwrStream_t, Uplo, std::size_t, float,
                                               const wwr::wwrFloatComplex *, std::size_t,
                                               const wwr::wwrFloatComplex *, int, float, float *,
                                               int);
template Status la_heamv<wwr::wwrDoubleComplex>(wwr::wwrStream_t, Uplo, std::size_t, double,
                                                const wwr::wwrDoubleComplex *, std::size_t,
                                                const wwr::wwrDoubleComplex *, int, double,
                                                double *, int);

} // namespace calaman
