/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of aberth per supported type pair
 *
 * Implementation unit of calaman.aberth, paired with interface.cppm's `extern
 * template` list; the device side is instantiated in aberth.cu. The three lists
 * stay in step -- a pair added to one alone links against nothing.
 */

module calaman.aberth;

// Imports are not re-exported through the primary interface; the signatures
// below need these names.
import wwr.runtime_api;
import wwr.complex;

namespace calaman {

template Status aberth<float, wwr::wwrFloatComplex>(wwr::wwrStream_t, int, int, const float *,
                                                    wwr::wwrFloatComplex *, int *, float, int);
template Status aberth<double, wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, int, const double *,
                                                      wwr::wwrDoubleComplex *, int *, double, int);
template Status aberth<wwr::wwrFloatComplex, wwr::wwrFloatComplex>(wwr::wwrStream_t, int, int,
                                                                   const wwr::wwrFloatComplex *,
                                                                   wwr::wwrFloatComplex *, int *,
                                                                   float, int);
template Status aberth<wwr::wwrDoubleComplex, wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, int,
                                                                     const wwr::wwrDoubleComplex *,
                                                                     wwr::wwrDoubleComplex *, int *,
                                                                     double, int);

} // namespace calaman
