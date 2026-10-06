// lartg.cu
//
// The device-kernel half of calaman.lartg: one per-element functor that turns
// (f[k], g[k]) into the plane rotation (c[k], s[k], r[k]), launched through
// wwr.extension.parallel_for. There is no host driver -- the module function is
// a thin wrapper that forwards here -- so, like lacgv.cu, nothing but this
// kernel lives on the device path. Shared unchanged between both backends.
//
// The per-element math is lartg_scalar (lartg.cuh), the DIRECT PORT of
// reference ?lartg's safe-scaling algorithm that ?steqr's kernel shares. Its
// four thresholds are precision constants, computed once on the host in the
// launcher (lartg_thresholds) and handed to every thread rather than recomputed.
#include "lartg_bridge.h"

#include "lapack/lartg/lartg.cuh"

#include <extension/parallel_for/parallel_for.cuh>

#include <cstddef>

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
    T c, s, r;
    lartg_scalar(f_[k], g_[k], LartgThresholds<T>{safmin_, safmax_, rtmin_, rtmax_}, &c, &s, &r);
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

  // The reference's la_constants, in this precision (lartg.cuh).
  const auto th = lartg_thresholds<T>();
  const LartgFunctor<T> functor{f, g, c, s, r, th.safmin, th.safmax, th.rtmin, th.rtmax};
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
