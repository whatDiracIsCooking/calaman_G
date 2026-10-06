/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of lanst() per type
 *
 * Implementation unit of calaman.lanst, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::lanst, so it is
 * instantiated once here. Keep this list in step with lanst.cu's.
 */

module calaman.lanst;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import calaman.common; // MatrixNorm

namespace calaman {

template Status lanst<float>(wwr::wwrStream_t, MatrixNorm, std::size_t, const float *,
                             const float *, float *);
template Status lanst<double>(wwr::wwrStream_t, MatrixNorm, std::size_t, const double *,
                              const double *, double *);

} // namespace calaman
