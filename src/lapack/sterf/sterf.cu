// sterf.cu
//
// The device half of calaman.sterf: one single-thread kernel that runs
// sterf_serial (sterf.cuh), LAPACK's ?sterf, and stores its INFO. The whole
// root-free QL/QR iteration, the per-block scaling and the closing sort run in
// that one thread -- no rotations are accumulated, so there is nothing to fan
// out, the shape of calaman.lahqr. Shared unchanged between both backends.
#include "sterf_bridge.h"

#include "lapack/sterf/sterf.cuh"

#include <runtime.h>

namespace calaman::device {

namespace {

/// @brief [kernel] One thread: *info = sterf_serial(n, d, e)
template<typename T>
__global__ void sterf_kernel(const int n, T *const d, T *const e, int *const info) {
  *info = sterf_serial(n, d, e);
}

} // namespace

template<typename T>
void sterf(const wwr::wwrStream_t stream, const int n, T *const d, T *const e, int *const info) {
  sterf_kernel<T><<<1, 1, 0, stream>>>(n, d, e, info);
}

// One per supported type, matching sterf_bridge.h and interface.cppm's
// `extern template` list -- float and double.
template void sterf<float>(wwr::wwrStream_t, int, float *, float *, int *);
template void sterf<double>(wwr::wwrStream_t, int, double *, double *, int *);

} // namespace calaman::device
