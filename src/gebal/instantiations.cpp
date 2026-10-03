/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of gebal per supported type
 *
 * Implementation unit of calaman.gebal. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the driver again -- its body names the device launchers declared
 * only in the interface's global module fragment, so it must be instantiated
 * here, inside this library. The kernels it launches are instantiated separately,
 * in gebal.cu, as device code.
 *
 * This list and the .cu's launcher instantiations must stay in step -- a type
 * added here without the matching launchers there links against nothing.
 */

module calaman.gebal;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without these. The ComplexToRealType the bodies use is reachable
// through the interface's own imports, so only the signature types are named.
// The Status return type arrives through the interface's `export import
// calaman.error_handling`.
import wwr.runtime_api; // wwrStream_t, wwrError_t
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex

namespace calaman {

template Status gebal<float>(wwr::wwrStream_t, GebalJob, int, float *, int, int *, int *, float *,
                             int *, int);
template Status gebal<double>(wwr::wwrStream_t, GebalJob, int, double *, int, int *, int *,
                              double *, int *, int);
template Status gebal<wwr::wwrFloatComplex>(wwr::wwrStream_t, GebalJob, int, wwr::wwrFloatComplex *,
                                            int, int *, int *, float *, int *, int);
template Status gebal<wwr::wwrDoubleComplex>(wwr::wwrStream_t, GebalJob, int,
                                             wwr::wwrDoubleComplex *, int, int *, int *, double *,
                                             int *, int);

} // namespace calaman
