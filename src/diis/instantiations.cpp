/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of diis_bufferSize and
 *        diis_push_and_extrapolate
 *
 * Implementation unit of calaman.diis, paired with interface.cppm's
 * `extern template` list. The push names the device launcher declared only in
 * the interface's global module fragment, so it is instantiated here, inside
 * this library; diis.cu instantiates the same types as device code.
 */

module calaman.diis;

import std;
import wwr.blas;
import wwr.runtime_api;

namespace calaman {

template std::size_t diis_bufferSize<float>(const DiisState &);
template std::size_t diis_bufferSize<double>(const DiisState &);
template Status diis_push_and_extrapolate<float>(wwr::wwrblasHandle_t, wwr::wwrStream_t,
                                                 DiisState &, const float *, const float *,
                                                 float *, int *, void *, std::size_t, float);
template Status diis_push_and_extrapolate<double>(wwr::wwrblasHandle_t, wwr::wwrStream_t,
                                                  DiisState &, const double *, const double *,
                                                  double *, int *, void *, std::size_t, double);

} // namespace calaman
