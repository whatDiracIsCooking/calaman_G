/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of the larmm driver per supported type
 *
 * Implementation unit of calaman.larmm, paired with interface.cppm's `extern
 * template` list; must stay in step with larmm.cu's device instantiations.
 */

module calaman.larmm;

import wwr.runtime_api;

namespace calaman {

template Status larmm<float>(wwr::wwrStream_t, std::size_t, const float *, const float *,
                             const float *, float *);
template Status larmm<double>(wwr::wwrStream_t, std::size_t, const double *, const double *,
                              const double *, double *);

} // namespace calaman
