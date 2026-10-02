/**
 * @file instantiations.cpp
 * @brief The explicit instantiations of the four complex_cast operations per type
 *
 * Implementation unit of calaman.complex_cast. Pairs with the `extern template`
 * declarations in interface.cppm: together they keep every importer from
 * instantiating the wrappers again -- each body names the device launchers
 * (device::set_real_part etc.) declared only in the interface's global module
 * fragment, so it must be instantiated here, inside this library. The device-side
 * work it calls is instantiated separately, in complex_cast.cu, as device code.
 *
 * This list and the .cu's must stay in step -- a type added here without being
 * added there links against nothing.
 */

module calaman.complex_cast;

// An implementation unit implicitly imports its primary interface, but an import
// is not re-exported through it -- the names in the signatures below are not
// visible without these. ComplexToRealType is reachable through the interface's
// own imports, so only the concrete signature types are named here.
import std;              // std::size_t
import wwr.runtime_api; // wwrStream_t
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex

namespace calaman {

template void set_real_part<wwr::wwrFloatComplex>(wwr::wwrStream_t, wwr::wwrFloatComplex *,
                                                  const float *, std::size_t);
template void set_real_part<wwr::wwrDoubleComplex>(wwr::wwrStream_t, wwr::wwrDoubleComplex *,
                                                   const double *, std::size_t);

template void set_imag_part<wwr::wwrFloatComplex>(wwr::wwrStream_t, wwr::wwrFloatComplex *,
                                                  const float *, std::size_t);
template void set_imag_part<wwr::wwrDoubleComplex>(wwr::wwrStream_t, wwr::wwrDoubleComplex *,
                                                   const double *, std::size_t);

template void get_real_part<wwr::wwrFloatComplex>(wwr::wwrStream_t, float *,
                                                  const wwr::wwrFloatComplex *, std::size_t);
template void get_real_part<wwr::wwrDoubleComplex>(wwr::wwrStream_t, double *,
                                                   const wwr::wwrDoubleComplex *, std::size_t);

template void get_imag_part<wwr::wwrFloatComplex>(wwr::wwrStream_t, float *,
                                                  const wwr::wwrFloatComplex *, std::size_t);
template void get_imag_part<wwr::wwrDoubleComplex>(wwr::wwrStream_t, double *,
                                                   const wwr::wwrDoubleComplex *, std::size_t);

} // namespace calaman
