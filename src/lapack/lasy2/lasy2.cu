// lasy2.cu
//
// The device-kernel half of calaman.lasy2: a single-thread port of LAPACK's
// ?lasy2, launched through wwr.extension.parallel_for over a grid of one. The
// whole routine is O(1) -- it solves op(TL)*X + ISGN*X*op(TR) = SCALE*B for an
// X of order at most 2 -- so there is nothing to parallelise; one thread runs
// the entire reference algorithm, and the module wrapper is a thin front door.
//
// FAITHFUL TO THE REFERENCE, by design: the test oracle is reference ?lasy2, so
// the arithmetic, the complete-pivoting order, the SMIN perturbation and the
// scale/xnorm formulas all mirror dlasy2.f line for line. LAPACK's DLAMCH('P')
// (the relative machine precision) is FLT_EPSILON / DBL_EPSILON, and DLAMCH('S')
// (the safe minimum) is FLT_MIN / DBL_MIN -- the <cfloat> macros, which are plain
// constants usable on the device, where std::numeric_limits is not (its members
// are __host__ constexpr, rejected by nvcc without --expt-relaxed-constexpr).
// Indices are kept 1-based to track the Fortran; the column-major (i,j) accessors
// fold the leading dimension back out.
//
// REAL ONLY: LAPACK ships no complex ?lasy2 (the 2x2 complex Sylvester case is a
// different routine), so the surface is float / double, matching the reference.
// Shared unchanged between both backends: a .cu is compiled by the backend
// compiler, and parallel_for's launch machinery arrives through its device
// header.
#include "lasy2_bridge.h"

#include <extension/parallel_for/parallel_for.cuh>

#include <cfloat>
#include <cstddef>

namespace calaman::device {

namespace {

// abs / max in exact arithmetic (no rounding), so they match LAPACK's intrinsic
// ABS / MAX bit for bit -- the control flow (pivot choice, SMIN tests) must.
template<typename T>
__device__ __forceinline__ T dabs(const T v) {
  return v < T{0} ? -v : v;
}
template<typename T>
__device__ __forceinline__ T dmax(const T a, const T b) {
  return a > b ? a : b;
}

// LAPACK's DLAMCH constants, device-side: eps() is DLAMCH('P') (relative machine
// precision) and sfmin() is DLAMCH('S') (safe minimum). std::numeric_limits is a
// __host__ constexpr the device pass rejects, so these read the <cfloat> macros.
template<typename T>
struct lamch;
template<>
struct lamch<float> {
  static __device__ __forceinline__ float eps() { return FLT_EPSILON; }
  static __device__ __forceinline__ float sfmin() { return FLT_MIN; }
};
template<>
struct lamch<double> {
  static __device__ __forceinline__ double eps() { return DBL_EPSILON; }
  static __device__ __forceinline__ double sfmin() { return DBL_MIN; }
};

// The per-call solve. Members are const scalar / pointer, as device_functor
// requires: trivially copyable, and the const deletes the copy-assignment the
// grid-constant kernel copy would otherwise permit. One thread runs operator();
// the index is ignored (the grid is a single element).
template<typename T>
struct Lasy2Functor {
  const bool ltranl_;
  const bool ltranr_;
  const int isgn_;
  const int n1_;
  const int n2_;
  const T *const tl_;
  const int ldtl_;
  const T *const tr_;
  const int ldtr_;
  const T *const b_;
  const int ldb_;
  T *const scale_;
  T *const x_;
  const int ldx_;
  T *const xnorm_;
  int *const info_;

  __device__ void operator()(std::size_t) const {
    const T eps = lamch<T>::eps();            // DLAMCH('P')
    const T smlnum = lamch<T>::sfmin() / eps; // DLAMCH('S') / eps
    const T sgn = static_cast<T>(isgn_);
    const T one = T{1};
    const T two = T{2};
    const T half = T{0.5};
    const T eight = T{8};

    // Column-major, 1-based element accessors, as in the Fortran.
    const T *const tl = tl_;
    const int ldtl = ldtl_;
    const T *const tr = tr_;
    const int ldtr = ldtr_;
    const T *const b = b_;
    const int ldb = ldb_;
    T *const x = x_;
    const int ldx = ldx_;
    auto TL = [=](int i, int j) -> T { return tl[(i - 1) + (j - 1) * ldtl]; };
    auto TR = [=](int i, int j) -> T { return tr[(i - 1) + (j - 1) * ldtr]; };
    auto B = [=](int i, int j) -> T { return b[(i - 1) + (j - 1) * ldb]; };
    auto setX = [=](int i, int j, T v) { x[(i - 1) + (j - 1) * ldx] = v; };

    int info = 0;

    // Quick return: a degenerate order writes only info, like the reference.
    if (n1_ == 0 || n2_ == 0) {
      *info_ = 0;
      return;
    }

    const int k = n1_ + n1_ + n2_ - 2; // 1:1x1, 2:1x2, 3:2x1, 4:2x2

    if (k == 1) {
      // 1 by 1: TL11*X + SGN*X*TR11 = B11
      T tau1 = TL(1, 1) + sgn * TR(1, 1);
      T bet = dabs(tau1);
      if (bet <= smlnum) {
        tau1 = smlnum;
        bet = smlnum;
        info = 1;
      }
      T scale = one;
      const T gam = dabs(B(1, 1));
      if (smlnum * gam > bet) {
        scale = one / gam;
      }
      const T xv = (B(1, 1) * scale) / tau1;
      setX(1, 1, xv);
      *scale_ = scale;
      *xnorm_ = dabs(xv);
      *info_ = info;
      return;
    }

    if (k == 2 || k == 3) {
      // 1 by 2 (k==2) or 2 by 1 (k==3): build the 2x2 system, solve it below.
      T tmp[5]; // 1-based [1..4]
      T btmp[5];
      T smin;
      if (k == 2) {
        smin = dmax(eps * dmax(dmax(dabs(TL(1, 1)), dabs(TR(1, 1))),
                               dmax(dmax(dabs(TR(1, 2)), dabs(TR(2, 1))), dabs(TR(2, 2)))),
                    smlnum);
        tmp[1] = TL(1, 1) + sgn * TR(1, 1);
        tmp[4] = TL(1, 1) + sgn * TR(2, 2);
        if (ltranr_) {
          tmp[2] = sgn * TR(2, 1);
          tmp[3] = sgn * TR(1, 2);
        } else {
          tmp[2] = sgn * TR(1, 2);
          tmp[3] = sgn * TR(2, 1);
        }
        btmp[1] = B(1, 1);
        btmp[2] = B(1, 2);
      } else {
        smin = dmax(eps * dmax(dmax(dabs(TR(1, 1)), dabs(TL(1, 1))),
                               dmax(dmax(dabs(TL(1, 2)), dabs(TL(2, 1))), dabs(TL(2, 2)))),
                    smlnum);
        tmp[1] = TL(1, 1) + sgn * TR(1, 1);
        tmp[4] = TL(2, 2) + sgn * TR(1, 1);
        if (ltranl_) {
          tmp[2] = TL(1, 2);
          tmp[3] = TL(2, 1);
        } else {
          tmp[2] = TL(2, 1);
          tmp[3] = TL(1, 2);
        }
        btmp[1] = B(1, 1);
        btmp[2] = B(2, 1);
      }

      // Solve the 2x2 system by complete pivoting (the reference's label 40).
      const int locu12[5] = {0, 3, 4, 1, 2};
      const int locl21[5] = {0, 2, 1, 4, 3};
      const int locu22[5] = {0, 4, 3, 2, 1};
      const bool xswpiv[5] = {false, false, false, true, true};
      const bool bswpiv[5] = {false, false, true, false, true};

      // idamax over tmp[1..4]: first index of max magnitude (strict >).
      int ipiv = 1;
      {
        T dm = dabs(tmp[1]);
        for (int i = 2; i <= 4; ++i) {
          const T a = dabs(tmp[i]);
          if (a > dm) {
            dm = a;
            ipiv = i;
          }
        }
      }
      T u11 = tmp[ipiv];
      if (dabs(u11) <= smin) {
        info = 1;
        u11 = smin;
      }
      const T u12 = tmp[locu12[ipiv]];
      const T l21 = tmp[locl21[ipiv]] / u11;
      T u22 = tmp[locu22[ipiv]] - u12 * l21;
      const bool xswap = xswpiv[ipiv];
      const bool bswap = bswpiv[ipiv];
      if (dabs(u22) <= smin) {
        info = 1;
        u22 = smin;
      }
      if (bswap) {
        const T temp = btmp[2];
        btmp[2] = btmp[1] - l21 * temp;
        btmp[1] = temp;
      } else {
        btmp[2] = btmp[2] - l21 * btmp[1];
      }
      T scale = one;
      if ((two * smlnum) * dabs(btmp[2]) > dabs(u22) ||
          (two * smlnum) * dabs(btmp[1]) > dabs(u11)) {
        scale = half / dmax(dabs(btmp[1]), dabs(btmp[2]));
        btmp[1] *= scale;
        btmp[2] *= scale;
      }
      T x2[3]; // 1-based [1..2]
      x2[2] = btmp[2] / u22;
      x2[1] = btmp[1] / u11 - (u12 / u11) * x2[2];
      if (xswap) {
        const T temp = x2[2];
        x2[2] = x2[1];
        x2[1] = temp;
      }
      setX(1, 1, x2[1]);
      T xnorm;
      if (n1_ == 1) {
        setX(1, 2, x2[2]);
        xnorm = dabs(x2[1]) + dabs(x2[2]);
      } else {
        setX(2, 1, x2[2]);
        xnorm = dmax(dabs(x2[1]), dabs(x2[2]));
      }
      *scale_ = scale;
      *xnorm_ = xnorm;
      *info_ = info;
      return;
    }

    // 2 by 2 (k==4): solve the equivalent 4x4 system by complete pivoting.
    T smin = dmax(dmax(dabs(TR(1, 1)), dabs(TR(1, 2))), dmax(dabs(TR(2, 1)), dabs(TR(2, 2))));
    smin = dmax(smin,
                dmax(dmax(dabs(TL(1, 1)), dabs(TL(1, 2))), dmax(dabs(TL(2, 1)), dabs(TL(2, 2)))));
    smin = dmax(eps * smin, smlnum);

    T t16[5][5]; // 1-based [1..4][1..4]
    for (int i = 1; i <= 4; ++i) {
      for (int j = 1; j <= 4; ++j) {
        t16[i][j] = T{0};
      }
    }
    t16[1][1] = TL(1, 1) + sgn * TR(1, 1);
    t16[2][2] = TL(2, 2) + sgn * TR(1, 1);
    t16[3][3] = TL(1, 1) + sgn * TR(2, 2);
    t16[4][4] = TL(2, 2) + sgn * TR(2, 2);
    if (ltranl_) {
      t16[1][2] = TL(2, 1);
      t16[2][1] = TL(1, 2);
      t16[3][4] = TL(2, 1);
      t16[4][3] = TL(1, 2);
    } else {
      t16[1][2] = TL(1, 2);
      t16[2][1] = TL(2, 1);
      t16[3][4] = TL(1, 2);
      t16[4][3] = TL(2, 1);
    }
    if (ltranr_) {
      t16[1][3] = sgn * TR(1, 2);
      t16[2][4] = sgn * TR(1, 2);
      t16[3][1] = sgn * TR(2, 1);
      t16[4][2] = sgn * TR(2, 1);
    } else {
      t16[1][3] = sgn * TR(2, 1);
      t16[2][4] = sgn * TR(2, 1);
      t16[3][1] = sgn * TR(1, 2);
      t16[4][2] = sgn * TR(1, 2);
    }
    T btmp[5];
    btmp[1] = B(1, 1);
    btmp[2] = B(2, 1);
    btmp[3] = B(1, 2);
    btmp[4] = B(2, 2);

    // Gaussian elimination with complete pivoting. The pivot scan keeps the LAST
    // max in scan order (>=), as the reference does -- not idamax's first.
    int jpiv[5];
    for (int i = 1; i <= 3; ++i) {
      T xmax = T{0};
      int ipsv = i;
      int jpsv = i;
      for (int ip = i; ip <= 4; ++ip) {
        for (int jp = i; jp <= 4; ++jp) {
          if (dabs(t16[ip][jp]) >= xmax) {
            xmax = dabs(t16[ip][jp]);
            ipsv = ip;
            jpsv = jp;
          }
        }
      }
      if (ipsv != i) {
        for (int c = 1; c <= 4; ++c) {
          const T temp = t16[ipsv][c];
          t16[ipsv][c] = t16[i][c];
          t16[i][c] = temp;
        }
        const T temp = btmp[i];
        btmp[i] = btmp[ipsv];
        btmp[ipsv] = temp;
      }
      if (jpsv != i) {
        for (int r = 1; r <= 4; ++r) {
          const T temp = t16[r][jpsv];
          t16[r][jpsv] = t16[r][i];
          t16[r][i] = temp;
        }
      }
      jpiv[i] = jpsv;
      if (dabs(t16[i][i]) < smin) {
        info = 1;
        t16[i][i] = smin;
      }
      for (int j = i + 1; j <= 4; ++j) {
        t16[j][i] = t16[j][i] / t16[i][i];
        btmp[j] = btmp[j] - t16[j][i] * btmp[i];
        for (int c = i + 1; c <= 4; ++c) {
          t16[j][c] = t16[j][c] - t16[j][i] * t16[i][c];
        }
      }
    }
    if (dabs(t16[4][4]) < smin) {
      info = 1;
      t16[4][4] = smin;
    }
    T scale = one;
    if ((eight * smlnum) * dabs(btmp[1]) > dabs(t16[1][1]) ||
        (eight * smlnum) * dabs(btmp[2]) > dabs(t16[2][2]) ||
        (eight * smlnum) * dabs(btmp[3]) > dabs(t16[3][3]) ||
        (eight * smlnum) * dabs(btmp[4]) > dabs(t16[4][4])) {
      scale = (one / eight) /
              dmax(dmax(dabs(btmp[1]), dabs(btmp[2])), dmax(dabs(btmp[3]), dabs(btmp[4])));
      btmp[1] *= scale;
      btmp[2] *= scale;
      btmp[3] *= scale;
      btmp[4] *= scale;
    }
    T tmp[5];
    for (int i = 1; i <= 4; ++i) {
      const int kk = 5 - i;
      const T temp = one / t16[kk][kk];
      tmp[kk] = btmp[kk] * temp;
      for (int j = kk + 1; j <= 4; ++j) {
        tmp[kk] = tmp[kk] - (temp * t16[kk][j]) * tmp[j];
      }
    }
    for (int i = 1; i <= 3; ++i) {
      if (jpiv[4 - i] != 4 - i) {
        const T temp = tmp[4 - i];
        tmp[4 - i] = tmp[jpiv[4 - i]];
        tmp[jpiv[4 - i]] = temp;
      }
    }
    setX(1, 1, tmp[1]);
    setX(2, 1, tmp[2]);
    setX(1, 2, tmp[3]);
    setX(2, 2, tmp[4]);
    *scale_ = scale;
    *xnorm_ = dmax(dabs(tmp[1]) + dabs(tmp[3]), dabs(tmp[2]) + dabs(tmp[4]));
    *info_ = info;
  }
};

} // namespace

template<typename T>
void lasy2(const wwr::wwrStream_t stream, const bool ltranl, const bool ltranr, const int isgn,
           const int n1, const int n2, const T *const tl, const int ldtl, const T *const tr,
           const int ldtr, const T *const b, const int ldb, T *const scale, T *const x,
           const int ldx, T *const xnorm, int *const info) {
  const Lasy2Functor<T> functor{ltranl, ltranr, isgn, n1,    n2, tl,  ldtl,  tr,
                                ldtr,   b,      ldb,  scale, x,  ldx, xnorm, info};
  wwr::extension::parallel_for<std::size_t>(stream, std::size_t{1}, functor);
}

// One per supported precision, matching lasy2_bridge.h's declaration and
// interface.cppm's extern-template list -- a type added here without being added
// there, or vice versa, links against nothing.
template void lasy2<float>(wwr::wwrStream_t, bool, bool, int, int, int, const float *, int,
                           const float *, int, const float *, int, float *, float *, int, float *,
                           int *);
template void lasy2<double>(wwr::wwrStream_t, bool, bool, int, int, int, const double *, int,
                            const double *, int, const double *, int, double *, double *, int,
                            double *, int *);

} // namespace calaman::device
