/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of lanhf() per type
 *
 * Implementation unit of calaman.lanhf, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::lansf, so it is
 * instantiated once here. Every type must also be in lansf.cu's list.
 */

module calaman.lanhf;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex
import calaman.common; // MatrixNorm, Trans, Uplo

namespace calaman {

template Status lanhf<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, Trans, Uplo, std::size_t,
                                            const wwr::wwrFloatComplex *, float *);
template Status lanhf<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, Trans, Uplo, std::size_t,
                                             const wwr::wwrDoubleComplex *, double *);

} // namespace calaman
