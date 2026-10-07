/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of lanhp() per type
 *
 * Implementation unit of calaman.lanhp, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::lansp, so it is
 * instantiated once here. Every type must also be in lansp.cu's list.
 */

module calaman.lanhp;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex
import calaman.common; // MatrixNorm, Uplo

namespace calaman {

template Status lanhp<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t,
                                            const wwr::wwrFloatComplex *, float *);
template Status lanhp<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t,
                                             const wwr::wwrDoubleComplex *, double *);

} // namespace calaman
