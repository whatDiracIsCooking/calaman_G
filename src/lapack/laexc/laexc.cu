// laexc.cu
//
// The device-kernel half of calaman.laexc: a single-thread port of LAPACK's
// ?laexc, launched through wwr.extension.parallel_for over a grid of one. The
// routine swaps two adjacent diagonal blocks (each 1x1 or 2x2) of a real Schur
// form by an orthogonal similarity. The block arithmetic is O(1), but applying
// the resulting rotations/reflectors to T (and, when WANTQ, Q) touches O(N)
// rows/columns; one thread runs the entire reference algorithm, so there is no
// host round-trip and the module wrapper is a thin front door, the shape of
// calaman.lasy2.
//
// FAITHFUL TO THE REFERENCE, by design: the test oracle is reference ?laexc, so
// the structure -- the 1x1 Givens case, the small Sylvester solve, the length-3
// Householder reflectors, the max-norm rejection test against THRESH, and the
// ?lanv2 re-standardisation with its trailing rotations -- mirrors dlaexc.f line
// for line. The auxiliary routines it calls (DLARTG, DLASY2, DLARFG, DLANV2, the
// DLARFX reflector apply, DROT, DLANGE 'Max') are inlined here as __device__
// helpers so the whole swap is one kernel. LAPACK's DLAMCH('P') (relative
// machine precision) is FLT_EPSILON / DBL_EPSILON and DLAMCH('S') (safe minimum)
// is FLT_MIN / DBL_MIN -- the <cfloat> macros, plain constants usable on the
// device, where std::numeric_limits is not (its members are __host__ constexpr,
// rejected by nvcc). Indices track the Fortran 1-based scheme; the column-major
// (i,j) accessors fold the leading dimension out.
//
// REAL ONLY: LAPACK ships no complex ?laexc (a complex Schur form is already
// triangular, with no 2x2 blocks to swap), so the surface is float / double,
// matching the reference. Shared unchanged between both backends: a .cu is
// compiled by the backend compiler, and parallel_for's launch machinery plus the
// precision-neutral math (sqrt/fabs/hypot/copysign) arrive through device
// headers.
#include "laexc_bridge.h"

#include <extension/parallel_for/parallel_for.cuh>
#include <wrappers/math/math.cuh>

#include <cfloat>
#include <cstddef>

namespace calaman::device {

namespace {

// LAPACK's DLAMCH constants, device-side: eps() is DLAMCH('P') (relative machine
// precision), sfmin() is DLAMCH('S') (safe minimum). std::numeric_limits is a
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

// abs / max in exact arithmetic, matching LAPACK's ABS / MAX bit for bit so the
// control flow (pivot choice, SMIN tests, the THRESH rejection) matches.
template<typename T>
__device__ __forceinline__ T dabs(const T v) {
  return v < T{0} ? -v : v;
}
template<typename T>
__device__ __forceinline__ T dmax(const T a, const T b) {
  return a > b ? a : b;
}
template<typename T>
__device__ __forceinline__ T dmin(const T a, const T b) {
  return a < b ? a : b;
}

// ------------------------------------------------------------------ DLARTG
//
// One plane rotation (cs, sn, r) with cs^2 + sn^2 == 1 taking (f, g) to (r, 0),
// reference ?lartg's safe-scaled algorithm. The 1x1 swap below needs the full
// (cs, sn) pair; r is returned too though the caller discards it.
template<typename T>
__device__ void dlartg(const T f, const T g, T &cs, T &sn, T &r) {
  const T safmin = lamch<T>::sfmin();
  const T safmax = T{1} / safmin;
  const T rtmin = wwr::sqrt(safmin);
  const T rtmax = wwr::sqrt(safmax / T{2});
  const T f1 = wwr::fabs(f);
  const T g1 = wwr::fabs(g);
  if (g == T{0}) {
    cs = T{1};
    sn = T{0};
    r = f;
  } else if (f == T{0}) {
    cs = T{0};
    sn = wwr::copysign(T{1}, g);
    r = g1;
  } else if (f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax) {
    const T d = wwr::sqrt(f * f + g * g);
    cs = f1 / d;
    r = wwr::copysign(d, f);
    sn = g / r;
  } else {
    const T u = wwr::fmin(safmax, wwr::fmax(wwr::fmax(safmin, f1), g1));
    const T fs = f / u;
    const T gs = g / u;
    const T d = wwr::sqrt(fs * fs + gs * gs);
    cs = wwr::fabs(fs) / d;
    r = wwr::copysign(d, f);
    sn = gs / r;
    r = r * u;
  }
}

// ------------------------------------------------------------------ DROT
//
// Apply the plane rotation (cs, sn) to the pair of vectors (x, y), each of n
// elements with the given stride: x,y <- (cs*x + sn*y, cs*y - sn*x). This is the
// Level-1 BLAS DROT the reference uses to mix the two swapped rows/columns.
template<typename T>
__device__ void drot(const int n, T *x, const int incx, T *y, const int incy, const T cs,
                      const T sn) {
  int ix = 0, iy = 0;
  for (int i = 0; i < n; ++i) {
    const T tx = x[ix];
    const T ty = y[iy];
    x[ix] = cs * tx + sn * ty;
    y[iy] = cs * ty - sn * tx;
    ix += incx;
    iy += incy;
  }
}

// ------------------------------------------------------------------ DLANV2
//
// Schur-standardise the real 2x2 block [a b; c d], reference ?lanv2 verbatim (V.
// Sima's cancellation-reducing revision). a, b, c, d are updated in place and
// (cs, sn) returned so the caller can rotate the rest of T/Q; the eigenvalues
// ?lanv2 also emits are unused by ?laexc, so they are dropped.
template<typename T>
__device__ void dlanv2(T &a, T &b, T &c, T &d, T &cs, T &sn) {
  const T zero{0}, half{0.5}, one{1}, two{2}, multpl{4};
  const T eps = lamch<T>::eps();
  const T safmin = lamch<T>::sfmin();
  // safmn2 = base^(int(log(safmin/eps)/log(base)/2)); base is the radix (2).
  const T base = T{2};
  const T safmn2 = wwr::pow(base, static_cast<T>(static_cast<int>(
                                       wwr::log(safmin / eps) / wwr::log(base) / two)));
  const T safmx2 = one / safmn2;

  if (c == zero) {
    cs = one;
    sn = zero;
  } else if (b == zero) {
    cs = zero;
    sn = one;
    const T temp = d;
    d = a;
    a = temp;
    b = -c;
    c = zero;
  } else if ((a - d) == zero && wwr::copysign(one, b) != wwr::copysign(one, c)) {
    cs = one;
    sn = zero;
  } else {
    T temp = a - d;
    T p = half * temp;
    const T bcmax = dmax(dabs(b), dabs(c));
    const T bcmis = dmin(dabs(b), dabs(c)) * wwr::copysign(one, b) * wwr::copysign(one, c);
    T scale = dmax(dabs(p), bcmax);
    T z = (p / scale) * p + (bcmax / scale) * bcmis;
    if (z >= multpl * eps) {
      z = p + wwr::copysign(wwr::sqrt(scale) * wwr::sqrt(z), p);
      a = d + z;
      d = d - (bcmax / z) * bcmis;
      const T tau = wwr::hypot(c, z);
      cs = z / tau;
      sn = c / tau;
      b = b - c;
      c = zero;
    } else {
      T sigma = b + c;
      for (int count = 1;; ++count) {
        scale = dmax(dabs(temp), dabs(sigma));
        if (scale >= safmx2) {
          sigma = sigma * safmn2;
          temp = temp * safmn2;
          if (count <= 20) {
            continue;
          }
        }
        if (scale <= safmn2) {
          sigma = sigma * safmx2;
          temp = temp * safmx2;
          if (count <= 20) {
            continue;
          }
        }
        break;
      }
      p = half * temp;
      T tau = wwr::hypot(sigma, temp);
      cs = wwr::sqrt(half * (one + dabs(sigma) / tau));
      sn = -(p / (tau * cs)) * wwr::copysign(one, sigma);
      const T aa = a * cs + b * sn;
      const T bb = -a * sn + b * cs;
      const T cc = c * cs + d * sn;
      const T dd = -c * sn + d * cs;
      a = aa * cs + cc * sn;
      b = (bb * cs) + (dd * sn);
      c = -(aa * sn) + (cc * cs);
      d = -bb * sn + dd * cs;
      temp = half * (a + d);
      a = temp;
      d = temp;
      if (c != zero) {
        if (b != zero) {
          if (wwr::copysign(one, b) == wwr::copysign(one, c)) {
            const T sab = wwr::sqrt(dabs(b));
            const T sac = wwr::sqrt(dabs(c));
            p = wwr::copysign(sab * sac, c);
            tau = one / wwr::sqrt(dabs(b + c));
            a = temp + p;
            d = temp - p;
            b = b - c;
            c = zero;
            const T cs1 = sab * tau;
            const T sn1 = sac * tau;
            const T rot = cs * cs1 - sn * sn1;
            sn = cs * sn1 + sn * cs1;
            cs = rot;
          }
        } else {
          b = -c;
          c = zero;
          const T rot = cs;
          cs = -sn;
          sn = rot;
        }
      }
    }
  }
}

// ------------------------------------------------------------------ DLASY2
//
// Solve op(TL)*X + ISGN*X*op(TR) = SCALE*B for the N1-by-N2 matrix X (1<=N1,N2
// <=2), reference ?lasy2 verbatim: here always called with LTRANL=LTRANR=.FALSE.
// and ISGN=-1 (the ?laexc coupling). tl/tr/b/x are small column-major slabs with
// leading dimension LDD=4 as laid out by ?laexc's D buffer. Returns via scale/x.
template<typename T>
__device__ void dlasy2(const int n1, const int n2, const T *tl, const int ldtl, const T *tr,
                        const int ldtr, const T *b, const int ldb, T &scale, T *x, const int ldx) {
  const T eps = lamch<T>::eps();
  const T smlnum = lamch<T>::sfmin() / eps;
  const T sgn = T{-1}; // ISGN = -1 in ?laexc
  const T one = T{1};
  const T two = T{2};
  const T half = T{0.5};
  const T eight = T{8};

  auto TL = [=](int i, int j) -> T { return tl[(i - 1) + (j - 1) * ldtl]; };
  auto TR = [=](int i, int j) -> T { return tr[(i - 1) + (j - 1) * ldtr]; };
  auto B = [=](int i, int j) -> T { return b[(i - 1) + (j - 1) * ldb]; };
  auto setX = [=](int i, int j, T v) { x[(i - 1) + (j - 1) * ldx] = v; };

  const int k = n1 + n1 + n2 - 2; // 1:1x1, 2:1x2, 3:2x1, 4:2x2

  if (k == 1) {
    T tau1 = TL(1, 1) + sgn * TR(1, 1);
    T bet = dabs(tau1);
    if (bet <= smlnum) {
      tau1 = smlnum;
      bet = smlnum;
    }
    T sc = one;
    const T gam = dabs(B(1, 1));
    if (smlnum * gam > bet) {
      sc = one / gam;
    }
    const T xv = (B(1, 1) * sc) / tau1;
    setX(1, 1, xv);
    scale = sc;
    return;
  }

  if (k == 2 || k == 3) {
    T tmp[5];
    T btmp[5];
    T smin;
    if (k == 2) {
      smin = dmax(eps * dmax(dmax(dabs(TL(1, 1)), dabs(TR(1, 1))),
                             dmax(dmax(dabs(TR(1, 2)), dabs(TR(2, 1))), dabs(TR(2, 2)))),
                  smlnum);
      tmp[1] = TL(1, 1) + sgn * TR(1, 1);
      tmp[4] = TL(1, 1) + sgn * TR(2, 2);
      tmp[2] = sgn * TR(1, 2);
      tmp[3] = sgn * TR(2, 1);
      btmp[1] = B(1, 1);
      btmp[2] = B(1, 2);
    } else {
      smin = dmax(eps * dmax(dmax(dabs(TR(1, 1)), dabs(TL(1, 1))),
                             dmax(dmax(dabs(TL(1, 2)), dabs(TL(2, 1))), dabs(TL(2, 2)))),
                  smlnum);
      tmp[1] = TL(1, 1) + sgn * TR(1, 1);
      tmp[4] = TL(2, 2) + sgn * TR(1, 1);
      tmp[2] = TL(2, 1);
      tmp[3] = TL(1, 2);
      btmp[1] = B(1, 1);
      btmp[2] = B(2, 1);
    }
    const int locu12[5] = {0, 3, 4, 1, 2};
    const int locl21[5] = {0, 2, 1, 4, 3};
    const int locu22[5] = {0, 4, 3, 2, 1};
    const bool xswpiv[5] = {false, false, false, true, true};
    const bool bswpiv[5] = {false, false, true, false, true};
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
      u11 = smin;
    }
    const T u12 = tmp[locu12[ipiv]];
    const T l21 = tmp[locl21[ipiv]] / u11;
    T u22 = tmp[locu22[ipiv]] - u12 * l21;
    const bool xswap = xswpiv[ipiv];
    const bool bswap = bswpiv[ipiv];
    if (dabs(u22) <= smin) {
      u22 = smin;
    }
    if (bswap) {
      const T temp = btmp[2];
      btmp[2] = btmp[1] - l21 * temp;
      btmp[1] = temp;
    } else {
      btmp[2] = btmp[2] - l21 * btmp[1];
    }
    T sc = one;
    if ((two * smlnum) * dabs(btmp[2]) > dabs(u22) ||
        (two * smlnum) * dabs(btmp[1]) > dabs(u11)) {
      sc = half / dmax(dabs(btmp[1]), dabs(btmp[2]));
      btmp[1] *= sc;
      btmp[2] *= sc;
    }
    T x2[3];
    x2[2] = btmp[2] / u22;
    x2[1] = btmp[1] / u11 - (u12 / u11) * x2[2];
    if (xswap) {
      const T temp = x2[2];
      x2[2] = x2[1];
      x2[1] = temp;
    }
    setX(1, 1, x2[1]);
    if (n1 == 1) {
      setX(1, 2, x2[2]);
    } else {
      setX(2, 1, x2[2]);
    }
    scale = sc;
    return;
  }

  // 2 by 2 (k==4): solve the equivalent 4x4 system by complete pivoting.
  T smin = dmax(dmax(dabs(TR(1, 1)), dabs(TR(1, 2))), dmax(dabs(TR(2, 1)), dabs(TR(2, 2))));
  smin = dmax(smin, dmax(dmax(dabs(TL(1, 1)), dabs(TL(1, 2))), dmax(dabs(TL(2, 1)), dabs(TL(2, 2)))));
  smin = dmax(eps * smin, smlnum);
  T t16[5][5];
  for (int i = 1; i <= 4; ++i) {
    for (int j = 1; j <= 4; ++j) {
      t16[i][j] = T{0};
    }
  }
  t16[1][1] = TL(1, 1) + sgn * TR(1, 1);
  t16[2][2] = TL(2, 2) + sgn * TR(1, 1);
  t16[3][3] = TL(1, 1) + sgn * TR(2, 2);
  t16[4][4] = TL(2, 2) + sgn * TR(2, 2);
  t16[1][2] = TL(1, 2);
  t16[2][1] = TL(2, 1);
  t16[3][4] = TL(1, 2);
  t16[4][3] = TL(2, 1);
  t16[1][3] = sgn * TR(2, 1);
  t16[2][4] = sgn * TR(2, 1);
  t16[3][1] = sgn * TR(1, 2);
  t16[4][2] = sgn * TR(1, 2);
  T btmp[5];
  btmp[1] = B(1, 1);
  btmp[2] = B(2, 1);
  btmp[3] = B(1, 2);
  btmp[4] = B(2, 2);
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
    t16[4][4] = smin;
  }
  T sc = one;
  if ((eight * smlnum) * dabs(btmp[1]) > dabs(t16[1][1]) ||
      (eight * smlnum) * dabs(btmp[2]) > dabs(t16[2][2]) ||
      (eight * smlnum) * dabs(btmp[3]) > dabs(t16[3][3]) ||
      (eight * smlnum) * dabs(btmp[4]) > dabs(t16[4][4])) {
    sc = (one / eight) /
         dmax(dmax(dabs(btmp[1]), dabs(btmp[2])), dmax(dabs(btmp[3]), dabs(btmp[4])));
    btmp[1] *= sc;
    btmp[2] *= sc;
    btmp[3] *= sc;
    btmp[4] *= sc;
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
  scale = sc;
}

// ------------------------------------------------------------------ DLARFG
//
// Generate the order-3 Householder reflector H = I - tau*v*v^T with v=[1;v2;v3]
// such that H*[alpha;x1;x2] = [beta;0;0], reference ?larfg. ?laexc only ever
// builds n==3 reflectors. On entry (alpha, x1, x2) are the three input scalars;
// on return alpha holds beta, (x1, x2) hold the reflector tail v2, v3, and tau
// is returned. The caller then overwrites the leading entry with 1 (the unit v).
template<typename T>
__device__ void dlarfg3(T &alpha, T &x1, T &x2, T &tau) {
  const T xnorm = wwr::hypot(x1, x2);
  if (xnorm == T{0}) {
    tau = T{0};
    return;
  }
  const T r = wwr::hypot(alpha, xnorm);
  const T beta = alpha >= T{0} ? -r : r;
  tau = (beta - alpha) / beta;
  const T scal = T{1} / (alpha - beta);
  x1 *= scal;
  x2 *= scal;
  alpha = beta;
}

// ------------------------------------------------------------------ DLARFX apply
//
// Apply the length-3 reflector H = I - tau*v*v^T (v=[v0;v1;v2]) to an m-by-n
// matrix C from the left ('L': C <- H*C) or the right ('R': C <- C*H), reference
// ?larfx with m or n == 3. The matrix C is column-major with leading dimension
// ldc; m/n are the OTHER dimension (the reflector side is always 3). This is the
// general gemv/ger form, not ?larfx's unrolled special cases -- the numbers
// match to rounding, which the oracle tolerance covers.
template<typename T>
__device__ void dlarfx3_left(const int m, const int n, const T *v, const T tau, T *c,
                             const int ldc) {
  // C <- (I - tau*v*v^T) C : for each column j, w = tau * (v . C(:,j)); C(:,j) -= w*v.
  if (tau == T{0}) {
    return;
  }
  for (int j = 0; j < n; ++j) {
    T *cj = c + static_cast<std::ptrdiff_t>(j) * ldc;
    T w = T{0};
    for (int i = 0; i < m; ++i) {
      w += v[i] * cj[i];
    }
    w *= tau;
    for (int i = 0; i < m; ++i) {
      cj[i] -= w * v[i];
    }
  }
}
template<typename T>
__device__ void dlarfx3_right(const int m, const int n, const T *v, const T tau, T *c,
                              const int ldc) {
  // C <- C (I - tau*v*v^T) : for each row i, w = tau * (C(i,:) . v); C(i,:) -= w*v.
  if (tau == T{0}) {
    return;
  }
  for (int i = 0; i < m; ++i) {
    T w = T{0};
    for (int j = 0; j < n; ++j) {
      w += c[i + static_cast<std::ptrdiff_t>(j) * ldc] * v[j];
    }
    w *= tau;
    for (int j = 0; j < n; ++j) {
      c[i + static_cast<std::ptrdiff_t>(j) * ldc] -= w * v[j];
    }
  }
}

// The whole swap, run by one thread. Members are const scalar / pointer, as
// device_functor requires: trivially copyable, and the const deletes the
// copy-assignment the grid-constant kernel copy would otherwise permit. The
// index is ignored (the grid is a single element).
template<typename T>
struct LaexcFunctor {
  const bool wantq_;
  const int n_;
  T *const t_;
  const int ldt_;
  T *const q_;
  const int ldq_;
  const int j1_; // 1-based, as in the Fortran
  const int n1_;
  const int n2_;
  int *const info_;

  __device__ void operator()(std::size_t) const {
    const int n = n_;
    const int ldt = ldt_;
    const int ldq = ldq_;
    const int j1 = j1_;
    const int n1 = n1_;
    const int n2 = n2_;
    T *const t = t_;
    T *const q = q_;

    // Column-major 1-based accessors into T (and Q); &T(i,j) for the BLAS-style
    // row/column spans ?laexc passes to DROT / the reflector applies.
    auto Tidx = [=](int i, int j) -> std::ptrdiff_t {
      return (i - 1) + static_cast<std::ptrdiff_t>(j - 1) * ldt;
    };
    auto Qidx = [=](int i, int j) -> std::ptrdiff_t {
      return (i - 1) + static_cast<std::ptrdiff_t>(j - 1) * ldq;
    };
    auto Tref = [=](int i, int j) -> T & { return t[Tidx(i, j)]; };

    *info_ = 0;

    if (n == 0 || n1 == 0 || n2 == 0) {
      return;
    }
    if (j1 + n1 > n) {
      return;
    }

    const int j2 = j1 + 1;
    const int j3 = j1 + 2;
    const int j4 = j1 + 3;

    if (n1 == 1 && n2 == 1) {
      // Swap two 1x1 blocks by a Givens rotation.
      const T t11 = Tref(j1, j1);
      const T t22 = Tref(j2, j2);
      T cs, sn, r;
      dlartg(Tref(j1, j2), t22 - t11, cs, sn, r);
      if (j3 <= n) {
        drot(n - j1 - 1, &t[Tidx(j1, j3)], ldt, &t[Tidx(j2, j3)], ldt, cs, sn);
      }
      drot(j1 - 1, &t[Tidx(1, j1)], 1, &t[Tidx(1, j2)], 1, cs, sn);
      Tref(j1, j1) = t22;
      Tref(j2, j2) = t11;
      if (wantq_) {
        drot(n, &q[Qidx(1, j1)], 1, &q[Qidx(1, j2)], 1, cs, sn);
      }
      return;
    }

    // Swapping two blocks of order 1 or 2: at least one is 2x2.
    const int ldd = 4;
    const int ldx = 2;
    const int nd = n1 + n2;

    // D <- T(j1:j1+nd-1, j1:j1+nd-1) (DLACPY 'Full').
    T d[16]; // column-major, leading dimension LDD=4
    for (int jj = 0; jj < nd; ++jj) {
      for (int ii = 0; ii < nd; ++ii) {
        d[ii + jj * ldd] = Tref(j1 + ii, j1 + jj);
      }
    }
    // DNORM = DLANGE('Max', nd, nd, D): the largest magnitude entry.
    T dnorm = T{0};
    for (int jj = 0; jj < nd; ++jj) {
      for (int ii = 0; ii < nd; ++ii) {
        dnorm = dmax(dnorm, dabs(d[ii + jj * ldd]));
      }
    }
    const T eps = lamch<T>::eps();
    const T smlnum = lamch<T>::sfmin() / eps;
    const T thresh = dmax(T{10} * eps * dnorm, smlnum);

    // Solve T11*X - X*T22 = SCALE*T12 (ISGN=-1): TL=D(1,1), TR=D(n1+1,n1+1),
    // B=D(1,n1+1), all with leading dimension LDD.
    T scale;
    T x[4]; // column-major, leading dimension LDX=2
    dlasy2(n1, n2, &d[0], ldd, &d[(n1) + (n1) * ldd], ldd, &d[0 + (n1) * ldd], ldd, scale, x, ldx);
    auto X = [&](int i, int j) -> T { return x[(i - 1) + (j - 1) * ldx]; };

    const int k = n1 + n1 + n2 - 3; // 1:(1,2), 2:(2,1), 3:(2,2)

    T u1[3], u2[3], u[3];
    T tau, tau1, tau2;

    if (k == 1) {
      // N1=1, N2=2.
      u[0] = scale;
      u[1] = X(1, 1);
      u[2] = X(1, 2);
      dlarfg3(u[2], u[0], u[1], tau);
      u[2] = T{1};
      const T t11 = Tref(j1, j1);
      dlarfx3_left(3, 3, u, tau, &d[0], ldd);
      dlarfx3_right(3, 3, u, tau, &d[0], ldd);
      if (dmax(dmax(dabs(d[2 + 0 * ldd]), dabs(d[2 + 1 * ldd])), dabs(d[2 + 2 * ldd] - t11)) >
          thresh) {
        *info_ = 1;
        return;
      }
      dlarfx3_left(3, n - j1 + 1, u, tau, &t[Tidx(j1, j1)], ldt);
      dlarfx3_right(j2, 3, u, tau, &t[Tidx(1, j1)], ldt);
      Tref(j3, j1) = T{0};
      Tref(j3, j2) = T{0};
      Tref(j3, j3) = t11;
      if (wantq_) {
        dlarfx3_right(n, 3, u, tau, &q[Qidx(1, j1)], ldq);
      }
    } else if (k == 2) {
      // N1=2, N2=1.
      u[0] = -X(1, 1);
      u[1] = -X(2, 1);
      u[2] = scale;
      dlarfg3(u[0], u[1], u[2], tau);
      u[0] = T{1};
      const T t33 = Tref(j3, j3);
      dlarfx3_left(3, 3, u, tau, &d[0], ldd);
      dlarfx3_right(3, 3, u, tau, &d[0], ldd);
      if (dmax(dmax(dabs(d[1 + 0 * ldd]), dabs(d[2 + 0 * ldd])), dabs(d[0 + 0 * ldd] - t33)) >
          thresh) {
        *info_ = 1;
        return;
      }
      dlarfx3_right(j3, 3, u, tau, &t[Tidx(1, j1)], ldt);
      dlarfx3_left(3, n - j1, u, tau, &t[Tidx(j1, j2)], ldt);
      Tref(j1, j1) = t33;
      Tref(j2, j1) = T{0};
      Tref(j3, j1) = T{0};
      if (wantq_) {
        dlarfx3_right(n, 3, u, tau, &q[Qidx(1, j1)], ldq);
      }
    } else {
      // N1=2, N2=2.
      u1[0] = -X(1, 1);
      u1[1] = -X(2, 1);
      u1[2] = scale;
      dlarfg3(u1[0], u1[1], u1[2], tau1);
      u1[0] = T{1};
      const T temp = -tau1 * (X(1, 2) + u1[1] * X(2, 2));
      u2[0] = -temp * u1[1] - X(2, 2);
      u2[1] = -temp * u1[2];
      u2[2] = scale;
      dlarfg3(u2[0], u2[1], u2[2], tau2);
      u2[0] = T{1};
      dlarfx3_left(3, 4, u1, tau1, &d[0], ldd);
      dlarfx3_right(4, 3, u1, tau1, &d[0], ldd);
      dlarfx3_left(3, 4, u2, tau2, &d[1], ldd);          // D(2,1)
      dlarfx3_right(4, 3, u2, tau2, &d[0 + 1 * ldd], ldd); // D(1,2)
      if (dmax(dmax(dabs(d[2 + 0 * ldd]), dabs(d[2 + 1 * ldd])),
               dmax(dabs(d[3 + 0 * ldd]), dabs(d[3 + 1 * ldd]))) > thresh) {
        *info_ = 1;
        return;
      }
      dlarfx3_left(3, n - j1 + 1, u1, tau1, &t[Tidx(j1, j1)], ldt);
      dlarfx3_right(j4, 3, u1, tau1, &t[Tidx(1, j1)], ldt);
      dlarfx3_left(3, n - j1 + 1, u2, tau2, &t[Tidx(j2, j1)], ldt);
      dlarfx3_right(j4, 3, u2, tau2, &t[Tidx(1, j2)], ldt);
      Tref(j3, j1) = T{0};
      Tref(j3, j2) = T{0};
      Tref(j4, j1) = T{0};
      Tref(j4, j2) = T{0};
      if (wantq_) {
        dlarfx3_right(n, 3, u1, tau1, &q[Qidx(1, j1)], ldq);
        dlarfx3_right(n, 3, u2, tau2, &q[Qidx(1, j2)], ldq);
      }
    }

    // Standardise any 2x2 block left in the swapped positions, with the trailing
    // DROT mixing the rest of its two rows/columns (and Q).
    if (n2 == 2) {
      T a = Tref(j1, j1), b = Tref(j1, j2), c = Tref(j2, j1), dd = Tref(j2, j2);
      T cs, sn;
      dlanv2(a, b, c, dd, cs, sn);
      Tref(j1, j1) = a;
      Tref(j1, j2) = b;
      Tref(j2, j1) = c;
      Tref(j2, j2) = dd;
      drot(n - j1 - 1, &t[Tidx(j1, j1 + 2)], ldt, &t[Tidx(j2, j1 + 2)], ldt, cs, sn);
      drot(j1 - 1, &t[Tidx(1, j1)], 1, &t[Tidx(1, j2)], 1, cs, sn);
      if (wantq_) {
        drot(n, &q[Qidx(1, j1)], 1, &q[Qidx(1, j2)], 1, cs, sn);
      }
    }

    if (n1 == 2) {
      const int jj3 = j1 + n2;
      const int jj4 = jj3 + 1;
      T a = Tref(jj3, jj3), b = Tref(jj3, jj4), c = Tref(jj4, jj3), dd = Tref(jj4, jj4);
      T cs, sn;
      dlanv2(a, b, c, dd, cs, sn);
      Tref(jj3, jj3) = a;
      Tref(jj3, jj4) = b;
      Tref(jj4, jj3) = c;
      Tref(jj4, jj4) = dd;
      if (jj3 + 2 <= n) {
        drot(n - jj3 - 1, &t[Tidx(jj3, jj3 + 2)], ldt, &t[Tidx(jj4, jj3 + 2)], ldt, cs, sn);
      }
      drot(jj3 - 1, &t[Tidx(1, jj3)], 1, &t[Tidx(1, jj4)], 1, cs, sn);
      if (wantq_) {
        drot(n, &q[Qidx(1, jj3)], 1, &q[Qidx(1, jj4)], 1, cs, sn);
      }
    }
  }
};

} // namespace

template<typename T>
void laexc(const wwr::wwrStream_t stream, const bool wantq, const int n, T *const t, const int ldt,
           T *const q, const int ldq, const int j1, const int n1, const int n2, int *const info) {
  const LaexcFunctor<T> functor{wantq, n, t, ldt, q, ldq, j1, n1, n2, info};
  wwr::extension::parallel_for<std::size_t>(stream, std::size_t{1}, functor);
}

// One per supported precision, matching laexc_bridge.h's declaration and
// interface.cppm's extern-template list -- a type added here without being added
// there, or vice versa, links against nothing.
template void laexc<float>(wwr::wwrStream_t, bool, int, float *, int, float *, int, int, int, int,
                           int *);
template void laexc<double>(wwr::wwrStream_t, bool, int, double *, int, double *, int, int, int, int,
                            int *);

} // namespace calaman::device
