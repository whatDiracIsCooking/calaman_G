/**
 * @file host_cu_canary.cu
 * @brief Host-side defects in launch wrappers inside a `.cu`
 *
 * Each defect is in host code nvcc (or clang -x hip) compiles as part of a
 * device library, the code CALAMAN_ENABLE_ASAN once left uninstrumented and
 * CALAMAN_ENABLE_UBSAN must reach through -Xcompiler. The kernel gives the
 * archive a real device pass; the sanitizer aborts before it launches.
 * Registered by this directory's CMakeLists.txt only under asan / hip-asan
 * (overflow) and ubsan / hip-ubsan (signed overflow). See docs/sanitizers.md
 * (G1, G4).
 */
#include "host_cu_canary.h"

#include <runtime.h>

#include <climits>

namespace calaman::canary {

namespace {

__global__ void fill(int *const out, const int value) {
  out[threadIdx.x] = value;
}

void launch_fill(const int value) {
  int *d_out = nullptr;
  (void)wwr::wwrMalloc(reinterpret_cast<void **>(&d_out), sizeof(int));
  fill<<<1, 1>>>(d_out, value);
  (void)wwr::wwrDeviceSynchronize();
  (void)wwr::wwrFree(d_out);
}

} // namespace

// Stages launch arguments into a heap array and writes one past its end. The
// volatile index keeps the compiler from proving the store out of bounds.
void cu_heap_buffer_overflow() {
  volatile int past_end = 4;
  int *const staged = new int[4];
  staged[past_end] = 1;
  launch_fill(staged[0]);
  delete[] staged;
}

// Computes a launch argument with a signed int overflow. The volatile operand
// keeps the add at run time.
void cu_signed_overflow() {
  volatile int top = INT_MAX;
  const int value = top + 1;
  launch_fill(value);
}

} // namespace calaman::canary
