/**
 * @file tri_index.cuh
 * @brief Device index maps for the triangular storage formats -- full (tr) and
 *        packed (tp) -- shared by the ?tpttr / ?trttp conversion kernels
 *
 * Header-only `__device__` functions. Every index is 0-based and every format
 * is column-major, as LAPACK lays them out; (i, j) names an element of the
 * n-by-n triangle selected by the `Upper` template argument.
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

} // namespace calaman::device
