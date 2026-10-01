// laqps.cu
//
// The device-kernel half of calaman.laqps: the deferred LAWN 176
// partial-norm downdate, the one elementwise-over-columns stage of the blocked,
// level-3 pivoted QR panel. Everything else in laqps is a host composition of
// wrapped BLAS (iamax / swap / gemv / gemm / nrm2) plus larfg; this is the only
// piece that is genuinely per-column device work, so like laqp2.cu it is
// launched through wwr.extension.parallel_for (a device-code header #included
// here, not a module). Shared unchanged between both backends, like lacpy.cu.
//
// The crucial difference from laqp2's downdate: there the trailing block is
// fully updated each step, so a degraded column is recomputed immediately. Here
// the block update is DEFERRED to one gemm after the whole panel, so a column
// flagged during an in-block step cannot be recomputed until after that gemm.
// The reference LAPACK threads the degraded columns onto a serial linked list
// (LSTICC, stored in VN2); this replaces that with a persistent device MASK:
// the kernel only ever RAISES flags[c] (writes 1), never clears it, and the host
// zeroes the whole mask once before step 1, so a column flagged on any step
// stays flagged until the host compacts the mask and recomputes those norms
// exactly after the panel gemm.
#include "laqps_bridge.h"

#include "extension/parallel_for/parallel_for.cuh"

#include <cmath>
#include <cstddef>

namespace calaman::device {

namespace {

/// @brief Per-column deferred-downdate functor invoked by parallel_for, index c
///
/// Trivially copyable with only const scalar members, as device_functor
/// requires (a mutable member would be UB in the grid-constant copy). operator()
/// is const-qualified and __device__; it writes only vn1_[c] (cheap path) or
/// raises flags_[c] (degraded path), never its own members. It never writes a 0
/// into flags_: a flag raised by an earlier step in the block must survive, so
/// clearing is the host's one-time job (laqps_clear_flags), not this functor's.
template<typename T>
struct DeferredDowndateFunctor {
  const T *const row_;  ///< A(rk, k+1): reflected row over trailing columns, stride lda_
  const std::size_t lda_;
  T *const vn1_;        ///< trailing tail vn1[k+1], stride 1 (cheap path shrinks in place)
  const T *const vn2_;  ///< trailing tail vn2[k+1], stride 1 (read only)
  int *const flags_;    ///< trailing tail flags[k+1], stride 1 (only ever raised to 1)
  const T tol3z_;

  __device__ void operator()(const std::size_t c) const {
    const T vn1 = vn1_[c];
    // A column already reduced to zero stays zero -- the ratio below would be
    // 0/0, and LAPACK's ?laqps guards the same way. Leave its flag as-is.
    if (vn1 == T{0}) {
      return;
    }

    // t = |A(rk, j)| / vn1[j]. The division never divides by zero (vn1 != 0).
    const T aij = row_[c * lda_];
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
    // is untrustworthy. Here the trailing column is not yet updated (the block
    // gemm is deferred), so the kernel cannot recompute -- it raises the
    // persistent flag and the host recomputes after the panel gemm. tol3z =
    // sqrt(eps) is the threshold; the ratio squared is the accumulated downdate.
    const T ratio = vn1 / vn2_[c];
    if (d * ratio * ratio <= tol3z_) {
      flags_[c] = 1;
      return;
    }

    // Cheap path: shrink the running partial norm; vn2 (the original) is left
    // untouched so later steps still test against it. The flag is left as the
    // host (or an earlier step) set it -- a column degraded earlier in the block
    // must stay flagged even if a later step's cheap test would pass.
    vn1_[c] = vn1 * std::sqrt(d);
  }
};

/// @brief Functor that zeroes one int slot -- the one-time flag clear
struct ClearFlagsFunctor {
  int *const flags_;
  __device__ void operator()(const std::size_t c) const { flags_[c] = 0; }
};

} // namespace

template<typename T>
void laqps_downdate(const wwr::wwrStream_t stream, const std::size_t count, const T *const row,
                    const std::size_t lda, T *const vn1, const T *const vn2, int *const flags,
                    const T tol3z) {
  if (count < 1) {
    return;
  }
  const DeferredDowndateFunctor<T> functor{row, lda, vn1, vn2, flags, tol3z};
  wwr::extension::parallel_for<std::size_t>(stream, count, functor);
}

void laqps_clear_flags(const wwr::wwrStream_t stream, const std::size_t count, int *const flags) {
  if (count < 1) {
    return;
  }
  const ClearFlagsFunctor functor{flags};
  wwr::extension::parallel_for<std::size_t>(stream, count, functor);
}

// One per supported type, matching laqps_bridge.h's declaration and the module's
// use sites -- float and double, as the panel is templated over.
template void laqps_downdate<float>(wwr::wwrStream_t, std::size_t, const float *, std::size_t,
                                    float *, const float *, int *, float);
template void laqps_downdate<double>(wwr::wwrStream_t, std::size_t, const double *, std::size_t,
                                     double *, const double *, int *, double);

} // namespace calaman::device
