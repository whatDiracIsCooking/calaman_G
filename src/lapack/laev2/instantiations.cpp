/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of the laev2 driver per supported type
 *
 * Implementation unit of calaman.laev2, paired with interface.cppm's `extern
 * template` list; must stay in step with laev2.cu's device instantiations.
 */

module calaman.laev2;

import wwr.runtime_api;

namespace calaman {

template Status laev2<float>(wwr::wwrStream_t, std::size_t, const float *, const float *,
                             const float *, float *, float *, float *, float *);
template Status laev2<double>(wwr::wwrStream_t, std::size_t, const double *, const double *,
                              const double *, double *, double *, double *, double *);

} // namespace calaman
