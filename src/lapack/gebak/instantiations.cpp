/**
 * @file instantiations.cpp
 * @brief The one explicit instantiation of gebak per supported type
 *
 * Implementation unit of calaman.gebak. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the driver again -- its body names the device launchers declared
 * only in the interface's global module fragment, so it must be instantiated here,
 * inside this library. The kernels it launches are instantiated separately, in
 * gebak.cu, as device code.
 *
 * This list and the .cu's launcher instantiations must stay in step -- a type
 * added here without the matching launchers there links against nothing.
 */

module calaman.gebak;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the complex element types named in the
// signatures below are not visible without this. ComplexToRealType the bodies use
// is reachable through the interface's own imports, and Status through its
// `export import calaman.error_handling`.
import wwr.complex; // wwrFloatComplex, wwrDoubleComplex

namespace calaman {

template Status gebak<float>(wwr::wwrStream_t, GebakJob, GebakSide, int, int, int, const float *,
                             int, float *, int);
template Status gebak<double>(wwr::wwrStream_t, GebakJob, GebakSide, int, int, int, const double *,
                              int, double *, int);
template Status gebak<wwr::wwrFloatComplex>(wwr::wwrStream_t, GebakJob, GebakSide, int, int, int,
                                            const float *, int, wwr::wwrFloatComplex *, int);
template Status gebak<wwr::wwrDoubleComplex>(wwr::wwrStream_t, GebakJob, GebakSide, int, int, int,
                                             const double *, int, wwr::wwrDoubleComplex *, int);

} // namespace calaman
