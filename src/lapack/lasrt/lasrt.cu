// lasrt.cu
//
// The device-kernel half of calaman.lasrt: sort a 1-D device array in place,
// LAPACK's ?lasrt. Declared in lasrt_bridge.h. There is no host driver -- the
// module function is a thin wrapper that forwards to this launcher, like
// calaman.set_element. Shared unchanged between both backends: under CUDA the
// .cu extension is all CMake needs, under HIP this directory's CMakeLists.txt
// forces LANGUAGE CXX back on so clang compiles it with -x hip.
//
// ONE SINGLE-THREAD KERNEL. The sort is inherently sequential (quicksort with an
// explicit stack, reverting to insertion sort on short ranges), and it branches
// on the data it is reordering at every step, so it runs in one thread rather
// than fanning out -- correct and simplest, the same "launch the obvious grid"
// call laset's triangular skip makes. A sorted array is a function of the input
// MULTISET and the direction alone, so this agrees with the reference ?lasrt bit
// for bit regardless of which comparison sort either uses; a parallel bitonic or
// merge sort is a possible later optimisation (README). lasrt is real-only (no
// ordering on complex), so T is float or double -- no c/z, unlike laset.
//
// The algorithm is LAPACK's dlasrt ported verbatim to 0-based inclusive ranges:
// median-of-3 pivot, Hoare partition, the larger sub-range pushed first so the
// 32-frame stack bounds n as LAPACK's STACK(2,32) does (n is int <= 2^31, well
// inside the ~2^32 that depth supports). The direction is a non-type template
// argument, so each specialization's comparisons resolve at compile time, the
// way laset's Region does.
#include "lasrt_bridge.h"

#include "common/enums.h"
#include <runtime.h>

namespace calaman::device {

namespace {

// Insertion sort reverts here for short ranges; the median-of-3 quicksort runs
// above it. LAPACK's SELECT.
constexpr int kInsertionThreshold = 20;

// LAPACK's STACK(2,32): with the larger sub-range pushed first, the pending
// stack depth is bounded by log2(n), so 32 frames admit n up to ~2^32 -- past
// every int n (<= 2^31 - 1). A frame is one inclusive [lo, hi] sub-range.
constexpr int kStackSize = 32;

/// @brief True iff the pair (a before b) is OUT of the requested sort order.
///
/// Increasing wants non-decreasing neighbours, so a > b is out of order;
/// decreasing wants a < b out of order. `Dir` is a template argument, so the
/// branch is resolved at compile time and the kernel below is comparison-only.
template<typename T, SortDir Dir>
__device__ bool out_of_order(const T a, const T b) {
  if constexpr (Dir == SortDir::I) {
    return a > b;
  } else {
    return a < b;
  }
}

/// @brief [kernel] Sort d[0..n-1] in place, in the order `Dir` selects
///
/// One thread runs LAPACK's dlasrt: pop an inclusive sub-range [start, endd]
/// from the explicit stack; insertion-sort it when it is short; otherwise
/// partition it about the median-of-3 pivot (a Hoare partition) and push the two
/// halves, the larger first. Entered only for n >= 2 (the launcher guards n <= 1).
template<typename T, SortDir Dir>
__global__ void lasrt_kernel(T *const d, const int n) {
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

      // Hoare partition about dmnmx. The two scans use the raw comparison (not
      // out_of_order): increasing pulls j down while d[j] > pivot and i up while
      // d[i] < pivot; decreasing mirrors it. A non-type template argument keeps
      // the inner tests branch-free per specialization.
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

      // Push both halves, the LARGER first, so the smaller is popped next and
      // the stack depth stays logarithmic -- LAPACK's ordering exactly.
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

} // namespace

template<typename T>
void lasrt(const wwr::wwrStream_t stream, const SortDir id, const int n, T *const d) {
  if (n <= 1) {
    return;
  }
  // One thread. Dispatch the runtime direction to the matching compile-time
  // specialization, as laset dispatches its Region.
  switch (id) {
  case SortDir::D:
    lasrt_kernel<T, SortDir::D><<<1, 1, 0, stream>>>(d, n);
    break;
  default:
    lasrt_kernel<T, SortDir::I><<<1, 1, 0, stream>>>(d, n);
    break;
  }
}

// One per supported type, matching lasrt_bridge.h's declaration and
// interface.cppm's extern-template list -- all three lists cover the same types.
// Real only: complex has no total order, so LAPACK ships no c/z ?lasrt.
template void lasrt<float>(wwr::wwrStream_t, SortDir, int, float *);
template void lasrt<double>(wwr::wwrStream_t, SortDir, int, double *);

} // namespace calaman::device
