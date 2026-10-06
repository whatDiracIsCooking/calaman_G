/**
 * @file tri_index.cuh
 * @brief Device index maps for the triangular storage formats -- full (tr),
 *        packed (tp) and Rectangular Full Packed (tf) -- shared by the
 *        ?tpttr / ?trttp / ?trttf / ?tfttr / ?tpttf / ?tfttp kernels
 *
 * Header-only `__device__` functions. Every index is 0-based and every format
 * is column-major, as LAPACK lays them out; (i, j) names an element of the
 * n-by-n triangle selected by the `Upper` template argument. The RFP layout is
 * described in src/lapack/trttf/README.md.
 *
 * Reached root-relative off the src/ root as "lapack/tri_index/tri_index.cuh";
 * link the INTERFACE target calaman::tri_index, which carries that root and
 * wwr.device. Device-only: include it from a .cu, never from a module interface.
 *
 * Usage:
 *   #include "lapack/tri_index/tri_index.cuh"
 *
 *   if (calaman::device::in_triangle<true>(i, j)) {
 *     a[i + j * lda] = ap[calaman::device::packed_index<true>(n, i, j)];
 *   }
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Whether (i, j) lies in the triangle, diagonal included
///
/// Upper is i <= j; lower is i >= j.
template<bool Upper>
__device__ __forceinline__ bool in_triangle(const std::size_t i, const std::size_t j) {
  if constexpr (Upper) {
    return i <= j;
  } else {
    return i >= j;
  }
}

/// @brief Offset of A(i, j) in the packed storage AP of an n-by-n triangle
///
/// Upper packs column j as AP[j(j+1)/2 .. j(j+1)/2 + j]; lower packs it as the
/// n - j elements from A(j, j) down.
/// @pre in_triangle<Upper>(i, j), i and j < n
template<bool Upper>
__device__ __forceinline__ std::size_t packed_index(const std::size_t n, const std::size_t i,
                                                    const std::size_t j) {
  if constexpr (Upper) {
    return i + j * (j + 1) / 2;
  } else {
    // j and 2n - j - 1 have opposite parity, so the product is even.
    return i + j * (2 * n - j - 1) / 2;
  }
}

/// @brief Where a triangle element lives in RFP storage
struct RfpSlot {
  std::size_t index; ///< Offset into ARF
  bool conj;         ///< Stored conjugated (the Hermitian convention); moot for real
};

/// @brief Slot of A(i, j) in the RFP array ARF of an n-by-n triangle
///
/// One map for all 8 TRANSR x UPLO x n-parity cases: @p transr selects the
/// (conjugate) transpose of the TRANSR = 'N' layout.
/// @pre in_triangle<Upper>(i, j), i and j < n
template<bool Upper>
__device__ __forceinline__ RfpSlot rfp_index(const bool transr, const std::size_t n,
                                             const std::size_t i, const std::size_t j) {
  // The 'N' layout is a column-major (n + even)-by-n2 array: one trapezoid in
  // place, the corner triangle (flip) stored as its conjugate transpose.
  const std::size_t n1 = n / 2;
  const std::size_t n2 = n - n1;
  const std::size_t even = 1 - n % 2;
  bool flip;
  std::size_t r;
  std::size_t c;
  if constexpr (Upper) {
    flip = j < n1;
    r = flip ? j + n1 + 1 : i;
    c = flip ? i : j - n1;
  } else {
    flip = j >= n2;
    r = flip ? j - n2 : i + even;
    c = flip ? i - n1 : j;
  }
  if (transr) {
    return {c + r * n2, !flip};
  }
  return {r + c * (n + even), flip};
}

} // namespace calaman::device
