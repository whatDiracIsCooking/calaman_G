/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of lanhe() per type
 *
 * Implementation unit of calaman.lanhe, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::lansy, so it is
 * instantiated once here. Every type must also be in lansy.cu's list.
 */

module calaman.lanhe;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex
import calaman.common; // MatrixNorm, Uplo

namespace calaman {

template Status lanhe<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t,
                                            const wwr::wwrFloatComplex *, std::size_t, float *);
template Status lanhe<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t,
                                             const wwr::wwrDoubleComplex *, std::size_t,
                                             double *);

} // namespace calaman
