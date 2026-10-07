/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of lacn2 / lacn2_bufferSize per type
 *
 * Implementation unit of calaman.lacn2, paired with the `extern template` list
 * in interface.cppm: the body names device launchers declared only in the
 * interface's global module fragment, so it is instantiated here, inside this
 * library. Keep this list, the interface's and lacn2.cu's in step.
 */

module calaman.lacn2;

import std;
import wwr.blas;

namespace calaman {

template std::size_t lacn2_bufferSize<float>(int);
template std::size_t lacn2_bufferSize<double>(int);
template Status lacn2<float>(wwr::wwrblasHandle_t, int, float *, float *, void *, std::size_t,
                             float &, int &, std::array<int, 3> &);
template Status lacn2<double>(wwr::wwrblasHandle_t, int, double *, double *, void *, std::size_t,
                              double &, int &, std::array<int, 3> &);

} // namespace calaman
