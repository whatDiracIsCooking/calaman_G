/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of lanhs() per type
 *
 * Implementation unit of calaman.lanhs, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::lanhs, so it is
 * instantiated once here. Keep this list in step with lanhs.cu's.
 */

module calaman.lanhs;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex
import calaman.common; // MatrixNorm

namespace calaman {

template Status lanhs<float>(wwr::wwrStream_t, MatrixNorm, std::size_t, const float *,
                             std::size_t, float *);
template Status lanhs<double>(wwr::wwrStream_t, MatrixNorm, std::size_t, const double *,
                              std::size_t, double *);
template Status lanhs<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                            const wwr::wwrFloatComplex *, std::size_t, float *);
template Status lanhs<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                             const wwr::wwrDoubleComplex *, std::size_t,
                                             double *);

} // namespace calaman
