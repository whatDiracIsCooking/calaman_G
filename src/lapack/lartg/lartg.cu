// lartg.cu
//
// The device-kernel half of calaman.lartg: one per-element functor that turns
// (f[k], g[k]) into the plane rotation (c[k], s[k], r[k]), launched through
// wwr.extension.parallel_for. There is no host driver -- the module function is
// a thin wrapper that forwards here -- so, like lacgv.cu, nothing but this
// kernel lives on the device path. Shared unchanged between both backends.
//
// A DIRECT PORT of reference ?lartg's safe-scaling algorithm (Anderson 2017,
// "Safe Scaling in the Level 1 BLAS"), reproduced per element so the oracle
// agrees: the g==0 / f==0 special cases, the unscaled fast path when both
// magnitudes sit inside [rtmin, rtmax], and the scaled fallback otherwise. The
// four thresholds are precision constants (safmin = smallest normal, safmax =
// 1/safmin, rtmin = sqrt(safmin), rtmax = sqrt(safmax/2)), computed once on the
// host in the launcher and handed to every thread rather than recomputed.
//
// NEUTRAL MATH, NEVER ::sqrtf vs ::sqrt: fabs/sqrt/copysign/fmin/fmax go through
// wwr.wrappers.math (wrappers/math/math.cuh), which spells the precision-
// divergent intrinsic once per type -- the same seam elem_ops forwards its
// transcendentals to. copysign gives LAPACK's SIGN(a, b) = |a| * sign(b); it is
// only reached where the sign source is non-zero, so IEEE's signed-zero rule
// never diverges from Fortran's.
#include "lartg_bridge.h"

#include <extension/parallel_for/parallel_for.cuh>
#include <wrappers/math/math.cuh>

#include <cstddef>
#include <cmath>
#include <limits>

namespace calaman::device {

namespace {

// The per-element rotation. Members are const scalar (pointers, the four
// thresholds), as device_functor requires: trivially copyable, and the const
// deletes the copy-assignment the grid-constant kernel copy would otherwise
// permit. f/g are read before c/s/r are written, so an output aliasing an input
// (e.g. r == f) is still correct, but the three outputs must be distinct.
template<typename T>
struct LartgFunctor {
  const T *const f_;
  const T *const g_;
  T *const c_;
  T *const s_;
  T *const r_;
  const T safmin_;
  const T safmax_;
  const T rtmin_;
  const T rtmax_;

  __device__ void operator()(const std::size_t k) const {
    const T f = f_[k];
    const T g = g_[k];
    const T f1 = wwr::fabs(f);
    const T g1 = wwr::fabs(g);

    T c, s, r;
    if (g == T{0}) {
      c = T{1};
      s = T{0};
      r = f;
    } else if (f == T{0}) {
      c = T{0};
      s = wwr::copysign(T{1}, g);
      r = g1;
    } else if (f1 > rtmin_ && f1 < rtmax_ && g1 > rtmin_ && g1 < rtmax_) {
      const T d = wwr::sqrt(f * f + g * g);
      c = f1 / d;
      r = wwr::copysign(d, f);
      s = g / r;
    } else {
      const T u = wwr::fmin(safmax_, wwr::fmax(wwr::fmax(safmin_, f1), g1));
      const T fs = f / u;
      const T gs = g / u;
      const T d = wwr::sqrt(fs * fs + gs * gs);
      c = wwr::fabs(fs) / d;
      r = wwr::copysign(d, f);
      s = gs / r;
      r = r * u;
    }

    c_[k] = c;
    s_[k] = s;
    r_[k] = r;
  }
};

} // namespace

template<typename T>
void lartg(const wwr::wwrStream_t stream, const std::size_t n, const T *f, const T *g, T *c, T *s,
           T *r) {
  if (n == 0) {
    return;
  }

  // The reference's la_constants, in this precision: safmin is the smallest
  // normal, safmax its reciprocal, and the two roots bound the fast path.
  const T safmin = std::numeric_limits<T>::min();
  const T safmax = T{1} / safmin;
  const T rtmin = std::sqrt(safmin);
  const T rtmax = std::sqrt(safmax / T{2});

  const LartgFunctor<T> functor{f, g, c, s, r, safmin, safmax, rtmin, rtmax};
  wwr::extension::parallel_for<std::size_t>(stream, n, functor);
}

// One per supported type, matching lartg_bridge.h's declaration and
// interface.cppm's extern-template list -- a type added here without being added
// there, or vice versa, links against nothing.
template void lartg<float>(wwr::wwrStream_t, std::size_t, const float *, const float *, float *,
                           float *, float *);
template void lartg<double>(wwr::wwrStream_t, std::size_t, const double *, const double *, double *,
                            double *, double *);

} // namespace calaman::device
