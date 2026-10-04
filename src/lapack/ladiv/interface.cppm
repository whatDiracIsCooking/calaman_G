/**
 * @file interface.cppm
 * @brief Primary interface for calaman.ladiv -- real-arithmetic complex division
 *        by Smith's algorithm, the device scalar helper for laln2 plus a batched
 *        device driver
 *
 * Two surfaces, one module. The SCALAR helper -- calaman::ladiv_scalar(a, b, c,
 * d, &p, &q), (p+i*q) = (a+i*b)/(c+i*d) by Smith's algorithm -- lives header-only
 * in "lapack/ladiv/ladiv.h" and is CLM_HOST_DEVICE, because its one real consumer is
 * laln2's KERNEL, which must #include it and call it per-thread (an imported host
 * function cannot be called from a __device__ context). That header is pulled
 * into the global module fragment below and re-#included by ladiv.cu; it is not a
 * module-exported name, since a kernel reaches it by include, not by import.
 *
 * The BATCHED driver -- calaman::ladiv(stream, n, a, b, c, d, p, q) -- is what
 * this module exports: n independent divisions launched on the device, the shape
 * the oracle test needs to exercise ladiv_scalar on a card (?ladiv is a scalar
 * routine, nothing to offload alone, so it is batched like calaman.lartg).
 * Enqueued on the given stream and returns WITHOUT synchronizing, like a BLAS
 * call; the caller synchronizes when it needs the outputs. All arrays are device
 * pointers the caller owns; nothing is allocated here, so a stream -- not a
 * device handle -- is the whole requirement.
 *
 * REAL ONLY. ?ladiv's complex-input variants (cladiv/zladiv) take two complex
 * operands and are a different signature; this is the real-arithmetic ?ladiv
 * (DLADIV/SLADIV) over float and double. `extern template` below pairs with
 * instantiations.cpp so an importer never re-instantiates the driver body, which
 * names the .cu-side launcher declared only in the GMF.
 *
 * Usage:
 *   import calaman.ladiv;     // also re-exports calaman::Status
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_a..d_d: length-n device inputs; d_p, d_q: length-n device outputs
 *   calaman::ladiv(stream, n, d_a, d_b, d_c, d_d, d_p, d_q);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the export import
// below supplies.
#include "error_handling/error_macros.h"

#include "ladiv_bridge.h"

export module calaman.ladiv;

import std;
import wwr.runtime_api;

// export import, not a plain import: ladiv RETURNS calaman::Status, so a consumer
// of `import calaman.ladiv;` must see Status's member functions, not just its
// name -- the same re-export calaman.lartg does.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace.

/// @brief Batch @p n complex divisions on @p stream (?ladiv, Smith's algorithm)
///
/// For each k in [0, n) writes p[k] + i*q[k] = (a[k]+i*b[k]) / (c[k]+i*d[k]) by
/// the header-only ladiv_scalar. Enqueues nothing and returns success when @p n
/// is 0. The two output arrays p, q must be distinct from each other; an output
/// may alias an input. Every divisor c[k] + i*d[k] must be non-zero. All arrays
/// live on @p stream's device.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param stream Stream the launch is enqueued on; the arrays live on its device
/// @param n Number of divisions (length of every array)
/// @param a Device input, numerator real parts
/// @param b Device input, numerator imaginary parts
/// @param c Device input, denominator real parts
/// @param d Device input, denominator imaginary parts
/// @param p Device output, quotient real parts
/// @param q Device output, quotient imaginary parts
/// @return Success, or the runtime error the kernel launch reported
export template<typename T>
Status ladiv(const wwr::wwrStream_t stream, const std::size_t n, const T *a, const T *b, const T *c,
             const T *d, T *p, T *q) {
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  device::ladiv(stream, n, a, b, c, d, p, q);
  // The launcher returns void, so the only way to catch a bad launch is the
  // runtime's sticky error -- checked the moment it is enqueued, as lartg does.
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status ladiv<float>(wwr::wwrStream_t, std::size_t, const float *, const float *,
                                    const float *, const float *, float *, float *);
extern template Status ladiv<double>(wwr::wwrStream_t, std::size_t, const double *, const double *,
                                     const double *, const double *, double *, double *);

} // namespace calaman
