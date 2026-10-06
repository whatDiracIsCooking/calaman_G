/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of laruv() per type
 *
 * Implementation unit of calaman.laruv, paired with interface.cppm's `extern
 * template` list: the wrapper names the GMF-only device::laruv, so it is
 * instantiated once here. Keep this list in step with laruv.cu's.
 */

module calaman.laruv;

// An implementation unit's implicit import of its interface does not re-export
// that interface's imports; the signatures below need this.
import wwr.runtime_api;

namespace calaman {

template Status laruv<float>(wwr::wwrStream_t, int *, int, float *);
template Status laruv<double>(wwr::wwrStream_t, int *, int, double *);

} // namespace calaman
