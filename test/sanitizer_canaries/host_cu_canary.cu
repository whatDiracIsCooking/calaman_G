/**
 * @file host_cu_canary.cu
 * @brief A host-side heap overflow in a launch wrapper inside a `.cu`
 *
 * The defect is in host code nvcc (or clang -x hip) compiles as part of a
 * device library, the code CALAMAN_ENABLE_ASAN once left uninstrumented. The
 * kernel gives the archive a real device pass; ASan aborts before it launches.
 * Registered by this directory's CMakeLists.txt only under asan / hip-asan.
 * See docs/sanitizers.md (G1).
 */
#include "host_cu_canary.h"

#include <runtime.h>

namespace calaman::canary {

namespace {

__global__ void fill(int *const out, const int value) {
  out[threadIdx.x] = value;
}

} // namespace

// Stages launch arguments into a heap array and writes one past its end. The
// volatile index keeps the compiler from proving the store out of bounds.
void cu_heap_buffer_overflow() {
  volatile int past_end = 4;
  int *const staged = new int[4];
  staged[past_end] = 1;
  int *d_out = nullptr;
  (void)wwr::wwrMalloc(reinterpret_cast<void **>(&d_out), sizeof(int));
  fill<<<1, 1>>>(d_out, staged[0]);
  (void)wwr::wwrDeviceSynchronize();
  (void)wwr::wwrFree(d_out);
  delete[] staged;
}

} // namespace calaman::canary
