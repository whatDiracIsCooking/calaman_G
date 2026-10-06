/**
 * @file host_device.h
 * @brief CLM_HOST_DEVICE: `__host__ __device__` in a device pass, empty otherwise
 *
 * For a helper that one body serves on both sides of the host/device line: a
 * device .cu calls it per-thread, while a host TU or a module's GMF, where the
 * attribute tokens are not keywords, sees a plain function. The test names the
 * vendors' device-compile passes (nvcc, HIP clang) without pulling a vendor
 * header. lapack/ladiv/ladiv.h carries an identical definition.
 *
 * Consumers #include this by its root-relative path, "common/host_device.h".
 */
#pragma once

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#define CLM_HOST_DEVICE __host__ __device__
#else
#define CLM_HOST_DEVICE
#endif
