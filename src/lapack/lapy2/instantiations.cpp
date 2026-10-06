/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of the lapy2 driver per supported type
 *
 * Implementation unit of calaman.lapy2, paired with interface.cppm's `extern
 * template` list; must stay in step with lapy2.cu's device instantiations.
 */

module calaman.lapy2;

import wwr.runtime_api;

namespace calaman {

template Status lapy2<float>(wwr::wwrStream_t, std::size_t, const float *, const float *, float *);
template Status lapy2<double>(wwr::wwrStream_t, std::size_t, const double *, const double *,
                              double *);

} // namespace calaman
