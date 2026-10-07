/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of langb() per type
 *
 * Implementation unit of calaman.langb, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::langb, so it is
 * instantiated once here. Keep this list in step with langb.cu's.
 */

module calaman.langb;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex
import calaman.common; // MatrixNorm

namespace calaman {

template Status langb<float>(wwr::wwrStream_t, MatrixNorm, std::size_t, std::size_t, std::size_t,
                             const float *, std::size_t, float *);
template Status langb<double>(wwr::wwrStream_t, MatrixNorm, std::size_t, std::size_t, std::size_t,
                              const double *, std::size_t, double *);
template Status langb<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, std::size_t, std::size_t,
                                            std::size_t, const wwr::wwrFloatComplex *, std::size_t,
                                            float *);
template Status langb<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, std::size_t, std::size_t,
                                             std::size_t, const wwr::wwrDoubleComplex *,
                                             std::size_t, double *);

} // namespace calaman
