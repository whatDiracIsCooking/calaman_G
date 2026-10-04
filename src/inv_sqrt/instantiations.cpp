/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of inv_sqrt and inv_sqrt_bufferSize
 *
 * Implementation unit of calaman.inv_sqrt. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the bodies again. The body of inv_sqrt names the device
 * launcher declared only in the interface's global module fragment, so it must
 * be instantiated here, inside this library; the device-side work it calls is
 * instantiated separately, in inv_sqrt.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing.
 */

module calaman.inv_sqrt;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without these.
import std;
import wwr.blas;
import wwr.solver;
import wwr.runtime_api;

namespace calaman {

template Status inv_sqrt_bufferSize<float>(wwr::wwrsolverDnHandle_t, const int, int *);
template Status inv_sqrt_bufferSize<double>(wwr::wwrsolverDnHandle_t, const int, int *);

template Status inv_sqrt<float>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, wwr::wwrStream_t,
                                const int, const float *, float *, float *, float *, float *,
                                float *, const int, int *, const float);
template Status inv_sqrt<double>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, wwr::wwrStream_t,
                                 const int, const double *, double *, double *, double *, double *,
                                 double *, const int, int *, const double);

} // namespace calaman
