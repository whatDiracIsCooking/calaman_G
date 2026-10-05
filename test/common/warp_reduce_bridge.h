/**
 * @file warp_reduce_bridge.h
 * @brief Plain-type launchers between warp_reduce_tests.cpp (host, GoogleTest)
 *        and warp_reduce_kernels.cu (device pass)
 *
 * Compiled on both sides of the host/device split, which on CUDA means two
 * standard libraries (nvcc's host pass is libstdc++, GoogleTest is libc++), so
 * the surface is builtin types only -- no std:: type crosses it. Each launcher
 * runs ONE kWarpSize-wide tile over host_in[0, warp_size()) and copies every
 * lane's result back into host_out[0, warp_size()), synchronously. Returns the
 * runtime's status code: 0 is success on both backends.
 */
#pragma once

namespace calaman::test {

/// @brief kWarpSize of the build: the length of every host_in / host_out
unsigned int warp_size();

/// @brief warp_reduce under AddOp
int warp_reduce_sum(const float *host_in, unsigned int nactive, float *host_out);

/// @brief warp_reduce under MaxNanOp
int warp_reduce_max_nan(const float *host_in, unsigned int nactive, float *host_out);

/// @brief warp_reduce under 2x2 matrix product mod kMat2Prime, the order probe
///
/// Each value packs a row-major [[a, b], [c, d]] as four 16-bit fields, a in
/// the high bits. Associative but not commutative, so a lane-order slip shows.
int warp_reduce_mat2(const unsigned long long *host_in, unsigned int nactive,
                     unsigned long long *host_out);

/// @brief The modulus warp_reduce_mat2 multiplies under: the largest 16-bit prime
inline constexpr unsigned long long kMat2Prime = 65521;

} // namespace calaman::test
