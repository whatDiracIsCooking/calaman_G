// laruv.cu
//
// The device-kernel half of calaman.laruv: one block of kLaruvMaxN threads runs
// laruv_block (laruv.cuh). Shared unchanged between both backends.
#include "laruv_bridge.h"

#include "lapack/laruv/laruv.cuh"

#include <runtime.h>

namespace calaman::device {

namespace {

/// @brief [kernel] ?laruv: every thread reads the seed, then the block draws.
template<typename T>
__global__ void laruv_kernel(int *const iseed, const int n, T *const x) {
  const int seed[4] = {iseed[0], iseed[1], iseed[2], iseed[3]};
  laruv_block(seed, n, x, iseed);
}

} // namespace

template<typename T>
void laruv(const wwr::wwrStream_t stream, int *const iseed, const int n, T *const x) {
  if (n < 1) {
    return;
  }
  laruv_kernel<T><<<1, kLaruvMaxN, 0, stream>>>(iseed, n, x);
}

template void laruv<float>(wwr::wwrStream_t, int *, int, float *);
template void laruv<double>(wwr::wwrStream_t, int *, int, double *);

} // namespace calaman::device
