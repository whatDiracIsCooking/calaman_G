/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of lansp() per type
 *
 * Implementation unit of calaman.lansp, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::lansp, so it is
 * instantiated once here. Keep this list in step with lansp.cu's.
 */

module calaman.lansp;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex
import calaman.common; // MatrixNorm, Uplo

namespace calaman {

template Status lansp<float>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t, const float *,
                             float *);
template Status lansp<double>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t, const double *,
                              double *);
template Status lansp<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t,
                                            const wwr::wwrFloatComplex *, float *);
template Status lansp<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t,
                                             const wwr::wwrDoubleComplex *, double *);

} // namespace calaman
