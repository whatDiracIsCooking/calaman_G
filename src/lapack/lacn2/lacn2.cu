// lacn2.cu
//
// The device-kernel half of calaman.lacn2: the five elementwise passes ?lacn2
// makes over its vectors (fill, unit vector, sign + record, sign comparison,
// alternating test vector), each a functor launched through
// wwr.extension.parallel_for. The reductions (?asum, i?amax) and the state
// machine stay host-side in interface.cppm.
//
// The sign test is ?lacn2's `X(I) .GE. ZERO`, so a NaN maps to -1 as in the
// reference (LAPACK >= 3.10; older releases used SIGN, which differs on -0).
#include "lacn2_bridge.h"

#include <extension/parallel_for/parallel_for.cuh>

#include <cstddef>

namespace calaman::device {

namespace {

template<typename T>
__device__ int sign_of(const T v) {
  return v >= T{0} ? 1 : -1;
}

template<typename T>
struct FillFunctor {
  T *const x_;
  const T value_;

  __device__ void operator()(const int i) const { x_[i] = value_; }
};

template<typename T>
struct UnitFunctor {
  T *const x_;
  const int j_;

  __device__ void operator()(const int i) const { x_[i] = i == j_ ? T{1} : T{0}; }
};

template<typename T>
struct SignFunctor {
  T *const x_;
  int *const isgn_;

  __device__ void operator()(const int i) const {
    const int s = sign_of(x_[i]);
    x_[i] = static_cast<T>(s);
    isgn_[i] = s;
  }
};

// Every differing thread stores the same 1, so the unsynchronized write is benign.
template<typename T>
struct SignChangedFunctor {
  const T *const x_;
  const int *const isgn_;
  int *const flag_;

  __device__ void operator()(const int i) const {
    if (sign_of(x_[i]) != isgn_[i]) {
      *flag_ = 1;
    }
  }
};

// ALTSGN*(ONE + DBLE(I-1)/DBLE(N-1)), evaluated in T as the reference does.
template<typename T>
struct AltsgnFunctor {
  T *const x_;
  const int n_;

  __device__ void operator()(const int i) const {
    const T mag = T{1} + static_cast<T>(i) / static_cast<T>(n_ - 1);
    x_[i] = (i % 2 == 0) ? mag : -mag;
  }
};

} // namespace

template<typename T>
void lacn2_fill(const wwr::wwrStream_t stream, const int n, const T value, T *const x) {
  wwr::extension::parallel_for<int>(stream, n, FillFunctor<T>{x, value});
}

template<typename T>
void lacn2_unit(const wwr::wwrStream_t stream, const int n, const int j, T *const x) {
  wwr::extension::parallel_for<int>(stream, n, UnitFunctor<T>{x, j});
}

template<typename T>
void lacn2_sign(const wwr::wwrStream_t stream, const int n, T *const x, int *const isgn) {
  wwr::extension::parallel_for<int>(stream, n, SignFunctor<T>{x, isgn});
}

template<typename T>
void lacn2_sign_changed(const wwr::wwrStream_t stream, const int n, const T *const x,
                        const int *const isgn, int *const flag) {
  wwr::extension::parallel_for<int>(stream, n, SignChangedFunctor<T>{x, isgn, flag});
}

template<typename T>
void lacn2_altsgn(const wwr::wwrStream_t stream, const int n, T *const x) {
  wwr::extension::parallel_for<int>(stream, n, AltsgnFunctor<T>{x, n});
}

// One per supported type, matching interface.cppm's extern-template list.
#define CLM_LACN2_INSTANTIATE(T)                                                                   \
  template void lacn2_fill<T>(wwr::wwrStream_t, int, T, T *);                                      \
  template void lacn2_unit<T>(wwr::wwrStream_t, int, int, T *);                                    \
  template void lacn2_sign<T>(wwr::wwrStream_t, int, T *, int *);                                  \
  template void lacn2_sign_changed<T>(wwr::wwrStream_t, int, const T *, const int *, int *);       \
  template void lacn2_altsgn<T>(wwr::wwrStream_t, int, T *);

CLM_LACN2_INSTANTIATE(float)
CLM_LACN2_INSTANTIATE(double)

#undef CLM_LACN2_INSTANTIATE

} // namespace calaman::device
