/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of sterf() per type
 *
 * Implementation unit of calaman.sterf, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::sterf, so it is
 * instantiated once here. Keep this list in step with sterf.cu's.
 */

module calaman.sterf;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need this.
import wwr.runtime_api;

namespace calaman {

template Status sterf<float>(wwr::wwrStream_t, int, float *, float *, int *);
template Status sterf<double>(wwr::wwrStream_t, int, double *, double *, int *);

} // namespace calaman
