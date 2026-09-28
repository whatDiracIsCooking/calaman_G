/**
 * @file elementwise_compare_bridge.h
 * @brief Device-launcher declarations shared between this util's interface unit
 *        and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by
 * elementwise_compare.cu directly. The declarations must live in the GMF, not
 * the module purview: a purview name gets module linkage and can never bind to
 * a definition compiled in a plain TU, which is what the .cu is.
 *
 * wwrStream_t arrives from the gpu* layer's include-only bridge header rather
 * than an `import`, since a GMF cannot import. It is the SAME type
 * wwr.runtime_api exports, so the wrapper passes its stream straight through.
 * Reading the backend define selected_backend.h needs is why the module links
 * wwr_backend PRIVATE -- see this directory's CMakeLists.txt.
 */

#pragma once

#include "extension/bridge/gpu_stream_bridge.h"

#include <cstddef>

namespace calaman::test::device {

/// @brief Stage 1 (exact): diffs[i] = (a[i] != b[i]) ? 1u : 0u for i in [0, n)
/// @tparam T Element type of the compared arrays; instantiated for float, double
template<typename T>
void write_mismatch_flags(wwr::wwrStream_t stream, const T *a, const T *b,
                          unsigned int *diffs, std::size_t n);

/// @brief Stage 1 (tolerance): flags[i] = (|a[i]-b[i]| > atol + rtol*|b[i]|) ? 1u : 0u
template<typename T>
void write_tolerance_flags(wwr::wwrStream_t stream, const T *a, const T *b, unsigned int *flags,
                           std::size_t n, T atol, T rtol);

/// @brief Stage 1 (magnitude): out[i] = |a[i] - b[i]|
template<typename T>
void write_abs_diff(wwr::wwrStream_t stream, const T *a, const T *b, T *out, std::size_t n);

/// @brief Stage 2 (count): single-block reduction summing diffs[0, n) into *result
void reduce_count(wwr::wwrStream_t stream, const unsigned int *diffs, unsigned int *result,
                  std::size_t n);

/// @brief Stage 2 (max): single-block reduction of the maximum of in[0, n) into *result
template<typename T>
void reduce_max(wwr::wwrStream_t stream, const T *in, T *result, std::size_t n);

} // namespace calaman::test::device
