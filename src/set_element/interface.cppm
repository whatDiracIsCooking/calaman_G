/**
 * @file interface.cppm
 * @brief Primary interface for calaman.set_element -- read one element of a
 *        device array by a device-held index
 *
 * Two device routines over a strided device vector, each one parallel_for item:
 *
 *     set_element:     d_result[0] <- d_x[(*d_idx - 1) * incx]        (gather)
 *     set_element_abs: d_result[0] <- |d_x[(*d_idx - 1) * incx]|      (magnitude)
 *
 * The index @p d_idx is a DEVICE pointer to a 1-based index -- the convention
 * I?AMAX writes. That is the whole reason this module exists: in a BLAS handle's
 * DEVICE pointer mode iamax writes its index to the device, and neither
 * cuBLAS nor hipBLAS has a call that then reads that element back (host-pointer
 * mode just syncs the index and lets the host do it -- see calaman.diff_norm's
 * ell_inf path). set_element_abs is the missing back half: iamax's index in,
 * the max-magnitude value out, entirely on the device. set_element is the same
 * read without the |.|, the generic "gather the element at a device-computed
 * index" primitive.
 *
 * Both are thin host wrappers forwarding to a device launcher in set_element.cu
 * (declared in set_element_bridge.h, included in the GMF), the same module/.cu
 * split calaman.complex_cast uses. A stream, not a handle, is the whole
 * requirement: one kernel, no allocation. Everything is enqueued on @p stream
 * and nothing synchronizes; the caller synchronizes when it needs d_result.
 *
 * The element type is constrained to calaman::usual_fp (float, double, and the two
 * complex types); the magnitude's real output type is calaman::ComplexToRealType<T>,
 * which is T itself for a real element. A valid in-range element is assumed to
 * exist -- an empty vector has no index to read, so the caller (as diff_norm
 * does for n == 0) must not call here in that case.
 *
 * Usage:
 *   import calaman.set_element;
 *   import wwr.runtime_api;   // wwrStream_t
 *   import wwr.blas;          // wwrblasGetStream, iamax in device pointer mode
 *   // d_x: device vector length n; d_idx: device int holding iamax's 1-based
 *   // index; d_mag: device scalar
 *   calaman::set_element_abs<double>(stream, d_x, 1, d_idx, d_mag);
 */

module;

#include "set_element_bridge.h"

export module calaman.set_element;

import std;
import wwr.runtime_api;     // wwrStream_t
import wwr.complex;         // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common;  // usual_fp, ComplexToRealType

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so each template carries its own
// `export` and the declarations below sit in the plain namespace.

/// @brief Gather d_x[(*d_idx - 1) * incx] into d_result[0] on @p stream
///
/// Reads the 1-based index @p d_idx holds on the device and copies that element
/// of @p d_x into @p d_result. Enqueued on @p stream; returns without
/// synchronizing. A valid in-range index is assumed to exist.
///
/// @tparam T Element type; one of the instantiated types (float, double, c, z)
/// @param stream Stream the launch is enqueued on; every pointer lives on its device
/// @param d_x Device source vector, stride @p incx
/// @param incx Stride between elements of @p d_x
/// @param d_idx Device pointer to a 1-based index (as iamax writes)
/// @param d_result Device scalar the selected element is written to
export template<calaman::usual_fp T>
void set_element(const wwr::wwrStream_t stream, const T *d_x, const int incx, const int *d_idx,
                 T *d_result) {
  device::set_element<T>(stream, d_x, incx, d_idx, d_result);
}

/// @brief Write |d_x[(*d_idx - 1) * incx]| into d_result[0] on @p stream
///
/// Like set_element, but writes the element's magnitude -- a real value, so
/// @p d_result is spelled calaman::ComplexToRealType<T> (T itself for a real
/// element). The device-pointer-mode companion to iamax. Enqueued on @p stream;
/// returns without synchronizing. A valid in-range index is assumed to exist.
///
/// @tparam T Element type; one of the instantiated types (float, double, c, z)
/// @param stream Stream the launch is enqueued on; every pointer lives on its device
/// @param d_x Device source vector, stride @p incx
/// @param incx Stride between elements of @p d_x
/// @param d_idx Device pointer to a 1-based index (as iamax writes)
/// @param d_result Device real scalar the magnitude is written to
export template<calaman::usual_fp T>
void set_element_abs(const wwr::wwrStream_t stream, const T *d_x, const int incx, const int *d_idx,
                     calaman::ComplexToRealType<T> *d_result) {
  device::set_element_abs<T, calaman::ComplexToRealType<T>>(stream, d_x, incx, d_idx, d_result);
}

extern template void set_element<float>(wwr::wwrStream_t, const float *, int, const int *, float *);
extern template void set_element<double>(wwr::wwrStream_t, const double *, int, const int *,
                                         double *);
extern template void set_element<wwr::wwrFloatComplex>(wwr::wwrStream_t, const wwr::wwrFloatComplex *,
                                                       int, const int *, wwr::wwrFloatComplex *);
extern template void set_element<wwr::wwrDoubleComplex>(wwr::wwrStream_t,
                                                        const wwr::wwrDoubleComplex *, int,
                                                        const int *, wwr::wwrDoubleComplex *);

extern template void set_element_abs<float>(wwr::wwrStream_t, const float *, int, const int *,
                                            float *);
extern template void set_element_abs<double>(wwr::wwrStream_t, const double *, int, const int *,
                                             double *);
extern template void set_element_abs<wwr::wwrFloatComplex>(wwr::wwrStream_t,
                                                           const wwr::wwrFloatComplex *, int,
                                                           const int *, float *);
extern template void set_element_abs<wwr::wwrDoubleComplex>(wwr::wwrStream_t,
                                                            const wwr::wwrDoubleComplex *, int,
                                                            const int *, double *);

} // namespace calaman
