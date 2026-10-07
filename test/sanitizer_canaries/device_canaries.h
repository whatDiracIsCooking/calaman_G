/**
 * @file device_canaries.h
 * @brief Host entry point to the compute-sanitizer canary kernels
 *
 * Declared here, defined in device_canaries.cu, called by device_main.cpp.
 */

#pragma once

namespace calaman::canary {

/// @brief Launch the canary kernel named @p name and synchronize
///
/// Names: memcheck, initcheck, racecheck-error, racecheck-warning, synccheck.
/// Runtime errors are ignored on purpose. @return false for an unknown name.
bool run_device_canary(const char *name);

} // namespace calaman::canary
