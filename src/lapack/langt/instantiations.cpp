/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of langt() per type
 *
 * Implementation unit of calaman.langt, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::langt, so it is
 * instantiated once here. Keep this list in step with langt.cu's.
 */

module calaman.langt;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex
import calaman.common; // MatrixNorm

namespace calaman {

template Status langt<float>(wwr::wwrStream_t, MatrixNorm, std::size_t, const float *,
                             const float *, const float *, float *);
template Status langt<double>(wwr::wwrStream_t, MatrixNorm, std::size_t, const double *,
                              const double *, const double *, double *);
template Status langt<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                            const wwr::wwrFloatComplex *,
                                            const wwr::wwrFloatComplex *,
                                            const wwr::wwrFloatComplex *, float *);
template Status langt<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                             const wwr::wwrDoubleComplex *,
                                             const wwr::wwrDoubleComplex *,
                                             const wwr::wwrDoubleComplex *, double *);

} // namespace calaman
