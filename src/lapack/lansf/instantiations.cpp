/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of lansf() per type
 *
 * Implementation unit of calaman.lansf, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::lansf, so it is
 * instantiated once here. Keep this list in step with lansf.cu's.
 */

module calaman.lansf;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need these.
import std;
import wwr.runtime_api;
import calaman.common; // MatrixNorm, Trans, Uplo

namespace calaman {

template Status lansf<float>(wwr::wwrStream_t, MatrixNorm, Trans, Uplo, std::size_t, const float *,
                             float *);
template Status lansf<double>(wwr::wwrStream_t, MatrixNorm, Trans, Uplo, std::size_t,
                              const double *, double *);

} // namespace calaman
