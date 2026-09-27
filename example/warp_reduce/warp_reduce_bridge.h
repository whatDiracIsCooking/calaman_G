/**
 * @file warp_reduce_bridge.h
 * @brief Declaration shared between this example's host TU and its device TU
 *
 * Included by main.cpp (an ordinary host compile) and by warp_reduce.cu (a
 * device pass), so it must compile in both modes -- which is what the
 * `_bridge` suffix marks. gpuStream_t therefore arrives by #include rather
 * than by import; see src/README.md ("The switch points").
 *
 * Usage:
 *   #include "warp_reduce_bridge.h"
 *
 *   wwr::example::warp_reduce_sum(stream.get(), n, input, output);
 */

#pragma once

#include "extension/bridge/gpu_stream_bridge.h"

#include <cstddef>

namespace wwr::example {

/// @brief Sum `[input, input + count)` into `output[0]`
///
/// Asynchronous, and a single block: `count` is unbounded by the launch
/// geometry but only one SM / CU is occupied. `count == 0` writes 0.
///
/// @param stream Stream to launch on
/// @param count Number of elements to reduce
/// @param input Device array of at least @p count elements, read only
/// @param output Device pointer to one writable element
void warp_reduce_sum(gpuStream_t stream, std::size_t count, const float *input, float *output);

} // namespace wwr::example
