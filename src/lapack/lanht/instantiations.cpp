/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of lanht() per type
 *
 * Implementation unit of calaman.lanht, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::lanst, so it is
 * instantiated once here. Every type must also be in lanst.cu's list.
 */

module calaman.lanht;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex
import calaman.common; // MatrixNorm

namespace calaman {

template Status lanht<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                            const float *, const wwr::wwrFloatComplex *, float *);
template Status lanht<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                             const double *, const wwr::wwrDoubleComplex *,
                                             double *);

} // namespace calaman
