/**
 * @file interface.cppm
 * @brief Primary interface for calaman.laev2 -- batched device ?laev2, the
 *        eigen-decomposition of a symmetric 2x2 matrix
 *
 * The scalar helper is device-only and header-only (laev2_scalar in
 * "lapack/sym2x2/sym2x2.cuh", reached by #include from a kernel, not by import).
 * This module exports the batched driver that exercises it on a card: n
 * independent ?laev2 calls, enqueued on a stream without synchronizing. All
 * arrays are caller-owned device pointers; nothing is allocated.
 *
 * Usage:
 *   import calaman.laev2;     // also re-exports calaman::Status
 *   import wwr.runtime_api;
 *   calaman::laev2(stream, n, d_a, d_b, d_c, d_rt1, d_rt2, d_cs1, d_sn1);
 */

module;

// CLM_TRY arrives by #include (a macro); needs calaman::Status, which the export
// import below supplies.
#include "error_handling/error_macros.h"

#include "laev2_bridge.h"

export module calaman.laev2;

import std;
import wwr.runtime_api;

export import calaman.error_handling; // Status -- the return type

namespace calaman {

/// @brief Batch @p n ?laev2 calls on @p stream: eigen-decompose [a b; b c]
///
/// For each k in [0, n) writes rt1[k], rt2[k] as ?lae2 does, and (cs1[k],
/// sn1[k]), the unit eigenvector of rt1[k]. Returns success without launching
/// when @p n is 0. The four outputs must be distinct; an output may alias an
/// input. All arrays live on @p stream's device.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @return Success, or the runtime error the kernel launch reported
export template<typename T>
Status laev2(const wwr::wwrStream_t stream, const std::size_t n, const T *a, const T *b, const T *c,
             T *rt1, T *rt2, T *cs1, T *sn1) {
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  device::laev2(stream, n, a, b, c, rt1, rt2, cs1, sn1);
  CLM_TRY(wwr::wwrGetLastError()); // the launcher is void; a bad launch is sticky
  return wwr::wwrSuccess;
}

extern template Status laev2<float>(wwr::wwrStream_t, std::size_t, const float *, const float *,
                                    const float *, float *, float *, float *, float *);
extern template Status laev2<double>(wwr::wwrStream_t, std::size_t, const double *, const double *,
                                     const double *, double *, double *, double *, double *);

} // namespace calaman
