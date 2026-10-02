/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of the two set_element operations per type
 *
 * Implementation unit of calaman.set_element. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the wrappers again -- each body names the device launchers
 * (device::set_element / device::set_element_abs) declared only in the
 * interface's global module fragment, so it must be instantiated here, inside
 * this library. The device-side work it calls is instantiated separately, in
 * set_element.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing.
 */

module calaman.set_element;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without these. ComplexToRealType is reachable through the interface's
// own imports, so only the concrete signature types are named here.
import wwr.runtime_api; // wwrStream_t
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex

namespace calaman {

template void set_element<float>(wwr::wwrStream_t, const float *, int, const int *, float *);
template void set_element<double>(wwr::wwrStream_t, const double *, int, const int *, double *);
template void set_element<wwr::wwrFloatComplex>(wwr::wwrStream_t, const wwr::wwrFloatComplex *, int,
                                                const int *, wwr::wwrFloatComplex *);
template void set_element<wwr::wwrDoubleComplex>(wwr::wwrStream_t, const wwr::wwrDoubleComplex *,
                                                 int, const int *, wwr::wwrDoubleComplex *);

template void set_element_abs<float>(wwr::wwrStream_t, const float *, int, const int *, float *);
template void set_element_abs<double>(wwr::wwrStream_t, const double *, int, const int *, double *);
template void set_element_abs<wwr::wwrFloatComplex>(wwr::wwrStream_t, const wwr::wwrFloatComplex *,
                                                    int, const int *, float *);
template void set_element_abs<wwr::wwrDoubleComplex>(wwr::wwrStream_t, const wwr::wwrDoubleComplex *,
                                                     int, const int *, double *);

} // namespace calaman
