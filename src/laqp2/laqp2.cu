// laqp2.cu
//
// The device-kernel half of calaman.laqp2: the LAWN 176 partial-norm
// downdate, the one elementwise-over-columns stage of the level-2 pivoted QR
// panel. Everything else in laqp2 is a host composition of wrapped BLAS plus
// larfg/larf; this is the only piece that is genuinely per-column device work,
// so it is launched through wwr.extension.parallel_for (a device-code header
// #included here, not a module) rather than hand-rolling a kernel. Shared
// unchanged between both backends, like lacpy.cu.
//
// One thread per trailing column j = i + 1 + k. Each thread reads the reflected
// row entry A(i, j) and the column's running partial norm vn1[j] / its original
// vn2[j], applies the Businger-Golub downdate, and either shrinks vn1[j] in
// place or raises flags[k] for the host to recompute vn1[j] from scratch. The
// recompute cannot happen here -- it is an nrm2 over a whole trailing column,
// not a per-column scalar step -- so the kernel only marks the columns and the
// host (laqp2.cppm) issues the exact norms after one sync of flags.
#include "laqp2_bridge.h"

#include "extension/parallel_for/parallel_for.cuh"

#include <cmath>
#include <cstddef>

namespace calaman::device {

namespace {

/// @brief Per-column downdate functor invoked by parallel_for for index k
///
/// Trivially copyable with only const scalar members, as device_functor
/// requires (a mutable member would be UB in the grid-constant copy). operator()
/// is const-qualified and __device__; it writes only vn1_[k] and flags_[k],
/// never its own members.
template<typename T>
struct DowndateFunctor {
  const T *const row_;  ///< A(i, i+1): reflected row over trailing columns, stride lda_
  const std::size_t lda_;
  T *const vn1_;        ///< trailing tail vn1[i+1], stride 1 (downdated in place)
  const T *const vn2_;  ///< trailing tail vn2[i+1], stride 1 (read only)
  int *const flags_;    ///< per-column recompute flag, stride 1
  const T tol3z_;

  __device__ void operator()(const std::size_t k) const {
    const T vn1 = vn1_[k];
    flags_[k] = 0;
    // A column already reduced to zero stays zero -- the ratio below would be
    // 0/0, and LAPACK's ?laqp2 guards the same way.
    if (vn1 == T{0}) {
      return;
    }

    // t = |A(i, j)| / vn1[j]. The division never divides by zero (vn1 != 0).
    const T aij = row_[k * lda_];
    const T abs_aij = aij < T{0} ? -aij : aij;
    const T t = abs_aij / vn1;

    // d = max(0, (1 - t)(1 + t)) = max(0, 1 - t^2): the squared shrink factor,
    // clamped so rounding cannot drive it negative under the sqrt.
    T d = (T{1} - t) * (T{1} + t);
    if (d < T{0}) {
      d = T{0};
    }

    // Relative-precision test (LAWN 176): if the downdated norm has lost too
    // much precision relative to the ORIGINAL norm vn2[j], the running estimate
    // is untrustworthy -- flag it for an exact recompute on the host. The ratio
    // squared is the accumulated downdate; tol3z = sqrt(eps) is the threshold.
    const T ratio = vn1 / vn2_[k];
    if (d * ratio * ratio <= tol3z_) {
      flags_[k] = 1;
      return;
    }

    // Cheap path: shrink the running partial norm; vn2 (the original) is left
    // untouched so later steps still test against it.
    vn1_[k] = vn1 * std::sqrt(d);
  }
};

} // namespace

template<typename T>
void laqp2_downdate(const wwr::wwrStream_t stream, const std::size_t count, const T *const row,
                    const std::size_t lda, T *const vn1, T *const vn2, int *const flags,
                    const T tol3z) {
  if (count < 1) {
    return;
  }
  // vn2 is read-only inside the functor; the const member takes a const pointer.
  const DowndateFunctor<T> functor{row, lda, vn1, vn2, flags, tol3z};
  wwr::extension::parallel_for<std::size_t>(stream, count, functor);
}

// One per supported type, matching laqp2_bridge.h's declaration and the module's
// use sites -- float and double, as the panel is templated over.
template void laqp2_downdate<float>(wwr::wwrStream_t, std::size_t, const float *, std::size_t,
                                    float *, float *, int *, float);
template void laqp2_downdate<double>(wwr::wwrStream_t, std::size_t, const double *, std::size_t,
                                     double *, double *, int *, double);

} // namespace calaman::device
