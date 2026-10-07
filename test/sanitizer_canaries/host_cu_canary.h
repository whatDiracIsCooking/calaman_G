/**
 * @file host_cu_canary.h
 * @brief Host entry point to the `.cu` host-side ASan canary
 *
 * Declared here, defined in host_cu_canary.cu, called by host_cu_main.cpp.
 */

#pragma once

namespace calaman::canary {

/// @brief Overflow a host heap buffer from inside a `.cu`, then launch a kernel
///
/// The overflow sits in the host pass of a device-library `.cu`, so only ASan
/// instrumentation of that pass can report it (docs/sanitizers.md, G1).
void cu_heap_buffer_overflow();

} // namespace calaman::canary
