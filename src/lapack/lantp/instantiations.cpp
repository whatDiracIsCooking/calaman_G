/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of lantp() per type
 *
 * Implementation unit of calaman.lantp, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::lantr, so it is
 * instantiated once here. Keep this list in step with lantr.cu's.
 */

module calaman.lantp;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex
import calaman.common; // MatrixNorm, Uplo, Diag

namespace calaman {

template Status lantp<float>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag, std::size_t, const float *,
                             float *);
template Status lantp<double>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag, std::size_t,
                              const double *, double *);
template Status lantp<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag, std::size_t,
                                            const wwr::wwrFloatComplex *, float *);
template Status lantp<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag,
                                             std::size_t, const wwr::wwrDoubleComplex *, double *);

} // namespace calaman
