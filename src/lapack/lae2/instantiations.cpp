/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of the lae2 driver per supported type
 *
 * Implementation unit of calaman.lae2, paired with interface.cppm's `extern
 * template` list; must stay in step with lae2.cu's device instantiations.
 */

module calaman.lae2;

import wwr.runtime_api;

namespace calaman {

template Status lae2<float>(wwr::wwrStream_t, std::size_t, const float *, const float *,
                            const float *, float *, float *);
template Status lae2<double>(wwr::wwrStream_t, std::size_t, const double *, const double *,
                             const double *, double *, double *);

} // namespace calaman
