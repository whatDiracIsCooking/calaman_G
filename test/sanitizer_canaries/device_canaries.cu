/**
 * @file device_canaries.cu
 * @brief Deliberately buggy kernels, one per compute-sanitizer tool
 *
 * Each kernel carries exactly one defect that its tool must report (the
 * memcheck leak is an unfreed allocation, no kernel); the launcher ignores
 * every runtime error, so the sanitizer is the only thing that can fail the
 * run. Registered by this directory's CMakeLists.txt only
 * under the compute-sanitizer preset, for the tool it selected.
 *
 * Every kernel lives in calaman:: on purpose: the racecheck/synccheck filter
 * (`--kernel-name kns=calaman`) must select them, so a canary that went green
 * would also catch the filter skipping calaman's kernels. See
 * docs/sanitizers.md (S2, G6).
 */
#include "device_canaries.h"

#include <runtime.h>

#include <cstring>

namespace calaman::canary {

namespace {

constexpr int kThreads = 64;

// memcheck: one thread writes one element past the end of the allocation.
__global__ void oob_global_write(int *const out, const int n) {
  if (threadIdx.x == 0) {
    out[n] = 1;
  }
}

// initcheck: reads a buffer nothing ever wrote.
__global__ void uninitialized_read(const int *const in, int *const out) {
  out[threadIdx.x] = in[threadIdx.x];
}

// racecheck, ERROR severity: threads of different warps write and read the
// same shared words with no barrier between.
__global__ void shared_race_cross_warp(int *const out) {
  __shared__ int s[kThreads];
  s[threadIdx.x] = static_cast<int>(threadIdx.x);
  out[threadIdx.x] = s[kThreads - 1 - threadIdx.x];
}

// racecheck, WARNING severity: the same race confined to one warp, which
// racecheck grades a warning (no __syncwarp between write and read).
__global__ void shared_race_intra_warp(int *const out) {
  __shared__ int s[32];
  s[threadIdx.x] = static_cast<int>(threadIdx.x);
  out[threadIdx.x] = s[31 - threadIdx.x];
}

// synccheck: only the first warp reaches the first barrier. The skipping warp
// must stay alive to the second one -- a thread that has exited counts as
// arrived, and synccheck then sees nothing wrong.
__global__ void divergent_barrier(int *const out) {
  if (threadIdx.x < kThreads / 2) {
    __syncthreads();
  }
  out[threadIdx.x] = static_cast<int>(threadIdx.x);
  __syncthreads();
}

} // namespace

bool run_device_canary(const char *const name) {
  int *a = nullptr;
  int *b = nullptr;
  (void)wwr::wwrMalloc(reinterpret_cast<void **>(&a), kThreads * sizeof(int));
  (void)wwr::wwrMalloc(reinterpret_cast<void **>(&b), kThreads * sizeof(int));

  bool known = true;
  if (std::strcmp(name, "memcheck-leak") == 0) {
    // memcheck --leak-check full: a device allocation never freed. Its
    // pointer is dropped here, so nothing below can release it.
    void *leaked = nullptr;
    (void)wwr::wwrMalloc(&leaked, kThreads * sizeof(int));
    leaked = nullptr;
  } else if (std::strcmp(name, "memcheck") == 0) {
    oob_global_write<<<1, kThreads>>>(a, kThreads);
  } else if (std::strcmp(name, "initcheck") == 0) {
    uninitialized_read<<<1, kThreads>>>(a, b);
  } else if (std::strcmp(name, "racecheck-error") == 0) {
    shared_race_cross_warp<<<1, kThreads>>>(a);
  } else if (std::strcmp(name, "racecheck-warning") == 0) {
    shared_race_intra_warp<<<1, 32>>>(a);
  } else if (std::strcmp(name, "synccheck") == 0) {
    divergent_barrier<<<1, kThreads>>>(a);
  } else {
    known = false;
  }

  (void)wwr::wwrDeviceSynchronize();
  (void)wwr::wwrFree(a);
  (void)wwr::wwrFree(b);
  return known;
}

} // namespace calaman::canary
