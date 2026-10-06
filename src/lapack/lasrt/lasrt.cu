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
// The algorithm (LAPACK's dlasrt, verbatim) lives in lasrt.cuh as
// lasrt_serial, so another kernel (?sterf) can run the same sort in-thread. The
// direction is a non-type template argument, so each specialization's
// comparisons resolve at compile time, the way laset's Region does.
#include "lasrt_bridge.h"

#include "lasrt.cuh"

#include "common/enums.h"
#include <runtime.h>

namespace calaman::device {

namespace {

/// @brief [kernel] Sort d[0..n-1] in place, in the order `Dir` selects
///
/// One thread runs lasrt_serial (lasrt.cuh), the shared in-thread body.
template<typename T, SortDir Dir>
__global__ void lasrt_kernel(T *const d, const int n) {
  lasrt_serial<T, Dir>(d, n);
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
