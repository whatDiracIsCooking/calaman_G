/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of lansb() per type
 *
 * Implementation unit of calaman.lansb, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::lansb, so it is
 * instantiated once here. Keep this list in step with lansb.cu's.
 */

module calaman.lansb;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex
import calaman.common; // MatrixNorm, Uplo

namespace calaman {

template Status lansb<float>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t, std::size_t,
                             const float *, std::size_t, float *);
template Status lansb<double>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t, std::size_t,
                              const double *, std::size_t, double *);
template Status lansb<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t,
                                            std::size_t, const wwr::wwrFloatComplex *, std::size_t,
                                            float *);
template Status lansb<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t,
                                             std::size_t, const wwr::wwrDoubleComplex *,
                                             std::size_t, double *);

} // namespace calaman
