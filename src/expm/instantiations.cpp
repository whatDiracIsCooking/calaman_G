/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of calaman.expm's templates, per type
 *
 * Implementation unit of calaman.expm. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating these bodies again -- each names the device launchers
 * (device::pade_even_odd / pade_split / expand_scale / abs_colsums / max_reduce)
 * declared only in the interface's global module fragment, so they must be
 * instantiated here, inside this library. The device-side work they call is
 * instantiated separately, in expm.cu, as device code.
 *
 * All four element types (float, double, and the two complex types). This list,
 * interface.cppm's extern-template list and expm.cu's device instantiations must
 * stay in step -- a type added in one place without the others links against
 * nothing.
 */

module calaman.expm;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without these.
import std;
import wwr.runtime_api;
import wwr.blas;
import wwr.solver;
import wwr.complex;
import wwr.wrappers.common;
import wwr.wrappers.blas;
import wwr.wrappers.solver;
import calaman.common;
import calaman.gebal;

namespace calaman {

// matrix_norm1
template wwr::ComplexToRealType<float> matrix_norm1<float>(wwr::wwrStream_t, int, const float *, int,
                                                           float *);
template wwr::ComplexToRealType<double> matrix_norm1<double>(wwr::wwrStream_t, int, const double *,
                                                             int, double *);
template wwr::ComplexToRealType<wwr::wwrFloatComplex>
matrix_norm1<wwr::wwrFloatComplex>(wwr::wwrStream_t, int, const wwr::wwrFloatComplex *, int,
                                   float *);
template wwr::ComplexToRealType<wwr::wwrDoubleComplex>
matrix_norm1<wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, const wwr::wwrDoubleComplex *, int,
                                    double *);

// pade_bufferSize
template wwr::wwrblasStatus_t pade_bufferSize<float>(wwr::wwrsolverDnHandle_t, int, int,
                                                     std::size_t *);
template wwr::wwrblasStatus_t pade_bufferSize<double>(wwr::wwrsolverDnHandle_t, int, int,
                                                      std::size_t *);
template wwr::wwrblasStatus_t pade_bufferSize<wwr::wwrFloatComplex>(wwr::wwrsolverDnHandle_t, int,
                                                                    int, std::size_t *);
template wwr::wwrblasStatus_t pade_bufferSize<wwr::wwrDoubleComplex>(wwr::wwrsolverDnHandle_t, int,
                                                                     int, std::size_t *);

// expm_bufferSize
template wwr::wwrblasStatus_t expm_bufferSize<float>(wwr::wwrsolverDnHandle_t, int, std::size_t *);
template wwr::wwrblasStatus_t expm_bufferSize<double>(wwr::wwrsolverDnHandle_t, int, std::size_t *);
template wwr::wwrblasStatus_t expm_bufferSize<wwr::wwrFloatComplex>(wwr::wwrsolverDnHandle_t, int,
                                                                    std::size_t *);
template wwr::wwrblasStatus_t expm_bufferSize<wwr::wwrDoubleComplex>(wwr::wwrsolverDnHandle_t, int,
                                                                     std::size_t *);

// pade
template wwr::wwrblasStatus_t pade<float>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                          wwr::wwrStream_t, int, int, const float *, int, float *,
                                          int, void *, std::size_t, int *);
template wwr::wwrblasStatus_t pade<double>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                           wwr::wwrStream_t, int, int, const double *, int, double *,
                                           int, void *, std::size_t, int *);
template wwr::wwrblasStatus_t pade<wwr::wwrFloatComplex>(wwr::wwrblasHandle_t,
                                                         wwr::wwrsolverDnHandle_t, wwr::wwrStream_t,
                                                         int, int, const wwr::wwrFloatComplex *, int,
                                                         wwr::wwrFloatComplex *, int, void *,
                                                         std::size_t, int *);
template wwr::wwrblasStatus_t
pade<wwr::wwrDoubleComplex>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, wwr::wwrStream_t, int,
                            int, const wwr::wwrDoubleComplex *, int, wwr::wwrDoubleComplex *, int,
                            void *, std::size_t, int *);

// expm
template wwr::wwrblasStatus_t expm<float>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                          wwr::wwrStream_t, int, const float *, int, float *, int,
                                          void *, std::size_t, int *, ExpmInfo *, ExpmOptions);
template wwr::wwrblasStatus_t expm<double>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                           wwr::wwrStream_t, int, const double *, int, double *, int,
                                           void *, std::size_t, int *, ExpmInfo *, ExpmOptions);
template wwr::wwrblasStatus_t expm<wwr::wwrFloatComplex>(
    wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, wwr::wwrStream_t, int,
    const wwr::wwrFloatComplex *, int, wwr::wwrFloatComplex *, int, void *, std::size_t, int *,
    ExpmInfo *, ExpmOptions);
template wwr::wwrblasStatus_t expm<wwr::wwrDoubleComplex>(
    wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, wwr::wwrStream_t, int,
    const wwr::wwrDoubleComplex *, int, wwr::wwrDoubleComplex *, int, void *, std::size_t, int *,
    ExpmInfo *, ExpmOptions);

} // namespace calaman
