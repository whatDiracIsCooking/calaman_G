/**
 * @file lasrt.cuh
 * @brief lasrt_serial: LAPACK's ?lasrt sort, run by one calling device thread
 *
 * The body of calaman.lasrt's kernel, as a `__device__` function another kernel
 * can call in-thread -- ?sterf sorts its eigenvalues this way, only when the
 * iteration converged. dlasrt ported verbatim to 0-based inclusive ranges:
 * median-of-3 pivot, Hoare partition, insertion sort at or below 20 elements,
 * the larger sub-range pushed first so LAPACK's 32-frame stack bounds every n.
 *
 * A downstream .cu reaches it root-relative as "lapack/lasrt/lasrt.cuh" by
 * linking calaman::lasrt::header (the src/ root). Device-only: include it from
 * a .cu, never from a module interface.
 *
 * Usage:
 *   #include "lapack/lasrt/lasrt.cuh"
 *   calaman::device::lasrt_serial<T, calaman::SortDir::I>(d, n);
 */

#pragma once

#include "common/enums.h"

namespace calaman::device {

namespace lasrt_detail {

/// LAPACK's SELECT: insertion sort below or at this range length.
inline constexpr int kInsertionThreshold = 20;
/// LAPACK's STACK(2,32); log2 depth bound with the larger half pushed first.
inline constexpr int kStackSize = 32;

/// @brief True iff a placed before b is OUT of the order `Dir` requests
template<typename T, SortDir Dir>
__device__ __forceinline__ bool out_of_order(const T a, const T b) {
  if constexpr (Dir == SortDir::I) {
    return a > b;
  } else {
    return a < b;
  }
}

} // namespace lasrt_detail

/// @brief Sort d[0..n-1] in place, in the order `Dir` selects (?lasrt)
///
/// Runs entirely in the calling thread; a no-op when @p n <= 1.
///
/// @tparam T Element type (float, double)
/// @tparam Dir SortDir::I (increasing) or SortDir::D (decreasing)
template<typename T, SortDir Dir>
__device__ void lasrt_serial(T *const d, const int n) {
  using lasrt_detail::kInsertionThreshold;
  using lasrt_detail::kStackSize;
  using lasrt_detail::out_of_order;
  if (n <= 1) {
    return;
  }
  int stack_lo[kStackSize];
  int stack_hi[kStackSize];
  int sp = 0;
  stack_lo[0] = 0;
  stack_hi[0] = n - 1;

  while (sp >= 0) {
    const int start = stack_lo[sp];
    const int endd = stack_hi[sp];
    --sp;

    if (endd - start <= kInsertionThreshold && endd - start > 0) {
      // Insertion sort: carry d[i] down past every out-of-order predecessor.
      for (int i = start + 1; i <= endd; ++i) {
        for (int j = i; j > start; --j) {
          if (out_of_order<T, Dir>(d[j - 1], d[j])) {
            const T tmp = d[j];
            d[j] = d[j - 1];
            d[j - 1] = tmp;
          } else {
            break;
          }
        }
      }
    } else if (endd - start > kInsertionThreshold) {
      // Median-of-3 pivot from the two ends and the midpoint.
      const T d1 = d[start];
      const T d2 = d[endd];
      const T d3 = d[(start + endd) / 2];
      T dmnmx;
      if (d1 < d2) {
        if (d3 < d1) {
          dmnmx = d1;
        } else if (d3 < d2) {
          dmnmx = d3;
        } else {
          dmnmx = d2;
        }
      } else {
        if (d3 < d2) {
          dmnmx = d2;
        } else if (d3 < d1) {
          dmnmx = d3;
        } else {
          dmnmx = d1;
        }
      }

      // Hoare partition about dmnmx; decreasing mirrors increasing.
      int i = start - 1;
      int j = endd + 1;
      for (;;) {
        if constexpr (Dir == SortDir::I) {
          do {
            --j;
          } while (d[j] > dmnmx);
          do {
            ++i;
          } while (d[i] < dmnmx);
        } else {
          do {
            --j;
          } while (d[j] < dmnmx);
          do {
            ++i;
          } while (d[i] > dmnmx);
        }
        if (i < j) {
          const T tmp = d[i];
          d[i] = d[j];
          d[j] = tmp;
        } else {
          break;
        }
      }

      // Push both halves, the LARGER first, so the stack depth stays
      // logarithmic -- LAPACK's ordering exactly.
      if (j - start > endd - j - 1) {
        ++sp;
        stack_lo[sp] = start;
        stack_hi[sp] = j;
        ++sp;
        stack_lo[sp] = j + 1;
        stack_hi[sp] = endd;
      } else {
        ++sp;
        stack_lo[sp] = j + 1;
        stack_hi[sp] = endd;
        ++sp;
        stack_lo[sp] = start;
        stack_hi[sp] = j;
      }
    }
    // endd - start <= 0: a single element or empty range, already sorted.
  }
}

} // namespace calaman::device
