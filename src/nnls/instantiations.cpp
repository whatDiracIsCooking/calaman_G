/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of nnls (and its buffer sizer) per type
 *
 * Implementation unit of calaman.nnls. Pairs with the `extern template`
 * declarations in nnls.cppm: together they keep every importer from
 * instantiating the solver again -- its body names the device launchers
 * declared only in the interface's global module fragment, so it is
 * instantiated here, inside this library. The device-side work it calls is
 * instantiated separately, in nnls.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing.
 */

module calaman.nnls;

// An implementation unit implicitly imports its primary interface, but that
// import is not re-exported through it -- the module-qualified names in the
// signatures below are not visible without these.
import std;
import wwr.blas;   // wwrblasHandle_t, wwrblasStatus_t
import wwr.solver; // wwrsolverDnHandle_t

namespace calaman {

template std::size_t nnls_bufferSize<float>(wwr::wwrsolverDnHandle_t, const int, const int);
template std::size_t nnls_bufferSize<double>(wwr::wwrsolverDnHandle_t, const int, const int);

template wwr::wwrblasStatus_t
nnls<float>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, const int, const int, const float *,
            const int, const float *, float *, void *, const std::size_t, const NnlsOptions<float> &,
            NnlsInfo<float> *);
template wwr::wwrblasStatus_t
nnls<double>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, const int, const int, const double *,
             const int, const double *, double *, void *, const std::size_t,
             const NnlsOptions<double> &, NnlsInfo<double> *);

} // namespace calaman
