/**
 * @file host_cu_canary.h
 * @brief Host entry points to the `.cu` host-side sanitizer canaries
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

/// @brief Overflow a signed int from inside a `.cu`, then launch a kernel
///
/// Only UBSan instrumentation of the `.cu`'s host pass can report it
/// (docs/sanitizers.md, G4).
void cu_signed_overflow();

} // namespace calaman::canary
