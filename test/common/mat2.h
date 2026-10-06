/**
 * @file mat2.h
 * @brief The reduction suites' order probe: 2x2 matrix product mod kMat2Prime
 *
 * Each value packs a row-major [[a, b], [c, d]] as four 16-bit fields, a in the
 * high bits. The product is associative but not commutative, so a reduction
 * that slips lane or thread order shows. Included by both the host tests (the
 * oracle) and the device kernels (the op), so mat2_mul is __host__ __device__
 * in a device pass and plain host code otherwise.
 */
#pragma once

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#define CLM_TEST_MAT2_HD __host__ __device__
#else
#define CLM_TEST_MAT2_HD
#endif

namespace calaman::test {

/// @brief The modulus mat2_mul multiplies under: the largest 16-bit prime
inline constexpr unsigned long long kMat2Prime = 65521;

/// @brief Field @p i (0..3, row-major) of packed matrix @p v
inline CLM_TEST_MAT2_HD unsigned long long mat2_at(const unsigned long long v, const int i) {
  return (v >> (48 - 16 * i)) & 0xFFFF;
}

/// @brief The packed product x * y mod kMat2Prime
inline CLM_TEST_MAT2_HD unsigned long long mat2_mul(const unsigned long long x,
                                                    const unsigned long long y) {
  const unsigned long long a =
      (mat2_at(x, 0) * mat2_at(y, 0) + mat2_at(x, 1) * mat2_at(y, 2)) % kMat2Prime;
  const unsigned long long b =
      (mat2_at(x, 0) * mat2_at(y, 1) + mat2_at(x, 1) * mat2_at(y, 3)) % kMat2Prime;
  const unsigned long long c =
      (mat2_at(x, 2) * mat2_at(y, 0) + mat2_at(x, 3) * mat2_at(y, 2)) % kMat2Prime;
  const unsigned long long d =
      (mat2_at(x, 2) * mat2_at(y, 1) + mat2_at(x, 3) * mat2_at(y, 3)) % kMat2Prime;
  return (a << 48) | (b << 32) | (c << 16) | d;
}

/// @brief Element @p i's matrix: distinct, small entries, so products stay mixed
inline unsigned long long mat2_of(const unsigned int i) {
  const unsigned long long a = 1 + i;
  const unsigned long long b = (3 * i) + 2;
  const unsigned long long c = (7 * i) + 5;
  const unsigned long long d = (11 * i) + 1;
  return (a << 48) | (b << 32) | (c << 16) | d;
}

} // namespace calaman::test

#undef CLM_TEST_MAT2_HD
