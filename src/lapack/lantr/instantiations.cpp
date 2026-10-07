/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of lantr() per type
 *
 * Implementation unit of calaman.lantr, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::lantr, so it is
 * instantiated once here. Keep this list in step with lantr.cu's.
 */

module calaman.lantr;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex
import calaman.common; // MatrixNorm, Uplo, Diag

namespace calaman {

template Status lantr<float>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag, std::size_t, std::size_t,
                             const float *, std::size_t, float *);
template Status lantr<double>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag, std::size_t, std::size_t,
                              const double *, std::size_t, double *);
template Status lantr<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag, std::size_t,
                                            std::size_t, const wwr::wwrFloatComplex *, std::size_t,
                                            float *);
template Status lantr<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag,
                                             std::size_t, std::size_t,
                                             const wwr::wwrDoubleComplex *, std::size_t, double *);

} // namespace calaman
