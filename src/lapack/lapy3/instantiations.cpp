/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of the lapy3 driver per supported type
 *
 * Implementation unit of calaman.lapy3, paired with interface.cppm's `extern
 * template` list; must stay in step with lapy3.cu's device instantiations.
 */

module calaman.lapy3;

import wwr.runtime_api;

namespace calaman {

template Status lapy3<float>(wwr::wwrStream_t, std::size_t, const float *, const float *,
                             const float *, float *);
template Status lapy3<double>(wwr::wwrStream_t, std::size_t, const double *, const double *,
                              const double *, double *);

} // namespace calaman
