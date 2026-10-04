// laqr2.cu
//
// The device-kernel half of calaman.laqr2: a single-thread port of LAPACK's
// ?laqr2, launched through wwr.extension.parallel_for over a grid of one. ?laqr2
// is aggressive early deflation (AED) for the real nonsymmetric eigensolver: it
// takes the trailing JW = min(NW, KBOT-KTOP+1) window of an upper-Hessenberg H,
// diagonalises it to real Schur form, deflates the eigenvalues whose coupling to
// the rest of H (the "spike", H(KWTOP,KWTOP-1)) is negligible, reflects the
// survivors back to Hessenberg form, and applies the accumulated window
// transform to H (and Z). The window work is O(JW^3) and the slab updates O(N*
// JW); one thread runs all of it with no host round-trip, the shape of
// calaman.laqr5 / calaman.lahqr.
//
// FAITHFUL TO THE REFERENCE, by design: the oracle is reference ?laqr2 (which
// calls ?lahqr for the window, not the recursive ?laqr4 -- that is ?laqr3), so
// the structure mirrors dlaqr2.f: the spike setup, the ?lahqr window Schur form,
// the spike-tip deflation test (|S*V(1,ns)| <= max(smlnum, ulp*foo)), the ?trexc
// reorder of the undeflatable eigenvalues, the graded-matrix bubble sort, the
// ?lanv2 eigenvalue read-back into SR/SI, and -- when any eigenvalue deflated or
// the spike vanished -- the ?larfg spike reflection, ?gehrd re-Hessenberg and
// ?ormhr / gemm slab updates. The auxiliaries (?lahqr, ?trexc over ?laexc,
// ?lanv2, ?larfg, ?gehrd, ?ormhr) are inlined here as __device__ helpers so the
// whole pass is one kernel; their arithmetic reproduces calaman.lahqr /
// calaman.laexc / calaman.trexc / calaman.gehrd / calaman.ormhr, which a host
// module staging through device memcpies cannot drive from inside a kernel.
//
// LAPACK's DLAMCH('S') (safe minimum) is FLT_MIN / DBL_MIN and DLAMCH('P'/'E')
// (eps) is FLT_EPSILON / DBL_EPSILON -- the <cfloat> macros, usable on the device
// where std::numeric_limits is not. Indices track the Fortran 1-based scheme; the
// column-major (i,j) accessors fold the leading dimension out. The reference's
// NH / NV slab-blocking is a GEMM-tiling choice; the single-thread kernel updates
// each slab unblocked, which changes only the accumulation order within tolerance.
//
// WINDOW-BOUNDED SCRATCH: the spike vector and the gehrd/ormhr reflectors live in
// on-stack buffers sized by kMaxWindow (the window JW, not N). A window larger
// than that is a configuration ?laqr2 is never driven with here (AED windows are
// small); the launcher clamps nothing, so the caller keeps NW <= kMaxWindow.
//
// REAL ONLY: ?laqr2 is the real path (float / double), matching the reference;
// a .cu is compiled by the backend compiler and parallel_for's launch machinery
// plus the precision-neutral math (sqrt/fabs/hypot/copysign) arrive through
// device headers, so the unit is shared unchanged between both backends.
#include "laqr2_bridge.h"

#include "extension/parallel_for/parallel_for.cuh"
#include "wrappers/math/math.cuh"

#include <cfloat>
#include <cstddef>

namespace calaman::device {

namespace {

// Largest deflation window the on-stack scratch holds. AED windows are small;
// NW beyond this is never used here, and the kernel does not clamp it.
constexpr int kMaxWindow = 192;

// LAPACK's DLAMCH constants, device-side: eps() is DLAMCH('P'/'E') (relative
// machine precision), sfmin() is DLAMCH('S') (safe minimum). std::numeric_limits
// is a __host__ constexpr the device pass rejects, so these read <cfloat>.
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

// ------------------------------------------------------------------ DROT
//
// Apply the plane rotation (cs, sn): x,y <- (cs*x + sn*y, cs*y - sn*x), the
// Level-1 BLAS DROT the Schur auxiliaries mix rows/columns with.
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

// ------------------------------------------------------------------ DLARTG
//
// One plane rotation (cs, sn, r) taking (f, g) to (r, 0), reference ?lartg's
// safe-scaled form. The 1x1 block swap needs the (cs, sn) pair.
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

// ------------------------------------------------------------------ DLANV2
//
// Schur-standardise the real 2x2 block [a b; c d], reference ?lanv2 verbatim (V.
// Sima's cancellation-reducing revision). a,b,c,d are updated in place,
// (rt1r,rt1i)/(rt2r,rt2i) receive the two eigenvalues, and (cs,sn) the rotation.
template<typename T>
__device__ void dlanv2(T &a, T &b, T &c, T &d, T &rt1r, T &rt1i, T &rt2r, T &rt2i, T &cs, T &sn) {
  const T zero{0}, half{0.5}, one{1}, two{2}, multpl{4};
  const T eps = lamch<T>::eps();
  const T safmin = lamch<T>::sfmin();
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
  rt1r = a;
  rt2r = d;
  if (c == zero) {
    rt1i = zero;
    rt2i = zero;
  } else {
    rt1i = wwr::sqrt(dabs(b)) * wwr::sqrt(dabs(c));
    rt2i = -rt1i;
  }
}

// The (cs, sn)-only overload ?laexc / ?trexc want (eigenvalues dropped).
template<typename T>
__device__ void dlanv2(T &a, T &b, T &c, T &d, T &cs, T &sn) {
  T r1r, r1i, r2r, r2i;
  dlanv2(a, b, c, d, r1r, r1i, r2r, r2i, cs, sn);
}

// ------------------------------------------------------------------ DLARFG
//
// Generate the order-n Householder reflector I - tau*v*v^T with v=[1; x] such
// that H*[alpha; x] = [beta; 0], reference ?larfg including the safmin rescale.
// On entry alpha and x[0..n-2] are the n input scalars (x unit stride); on return
// alpha holds beta, x the reflector tail, and tau is returned.
template<typename T>
__device__ T dlarfg(const int n, T &alpha, T *x, const int incx) {
  if (n <= 1) {
    return T{0};
  }
  const int m = n - 1;
  auto nrm2 = [&]() -> T {
    T s = T{0};
    for (int i = 0; i < m; ++i) {
      s = wwr::hypot(s, x[static_cast<std::ptrdiff_t>(i) * incx]);
    }
    return s;
  };
  T xnorm = nrm2();
  if (xnorm == T{0}) {
    return T{0};
  }
  T beta = -wwr::copysign(wwr::hypot(alpha, xnorm), alpha);
  const T safmin = lamch<T>::sfmin() / lamch<T>::eps();
  int knt = 0;
  if (dabs(beta) < safmin) {
    const T rsafmn = T{1} / safmin;
    do {
      ++knt;
      for (int i = 0; i < m; ++i) {
        x[static_cast<std::ptrdiff_t>(i) * incx] *= rsafmn;
      }
      beta *= rsafmn;
      alpha *= rsafmn;
    } while (dabs(beta) < safmin && knt < 20);
    xnorm = nrm2();
    beta = -wwr::copysign(wwr::hypot(alpha, xnorm), alpha);
  }
  const T tau = (beta - alpha) / beta;
  const T scal = T{1} / (alpha - beta);
  for (int i = 0; i < m; ++i) {
    x[static_cast<std::ptrdiff_t>(i) * incx] *= scal;
  }
  for (int j = 0; j < knt; ++j) {
    beta *= safmin;
  }
  alpha = beta;
  return tau;
}

// ------------------------------------------------------------------ DLARF (apply)
//
// Apply the reflector I - tau*v*v^T (v unit leading entry implied, i.e. v[0]==1)
// of length len to the m-by-n matrix C from the left ('L') or right ('R'),
// reference ?larf1f's semantics (v(1)=1 fused). The reflector side length is
// len; the other dimension is n (left) or m (right).
template<typename T>
__device__ void dlarf_left(const int len, const int n, const T *v, const T tau, T *c,
                           const int ldc) {
  if (tau == T{0}) {
    return;
  }
  for (int j = 0; j < n; ++j) {
    T *cj = c + static_cast<std::ptrdiff_t>(j) * ldc;
    T w = cj[0]; // v[0] == 1
    for (int i = 1; i < len; ++i) {
      w += v[i] * cj[i];
    }
    w *= tau;
    cj[0] -= w;
    for (int i = 1; i < len; ++i) {
      cj[i] -= w * v[i];
    }
  }
}
template<typename T>
__device__ void dlarf_right(const int m, const int len, const T *v, const T tau, T *c,
                            const int ldc) {
  if (tau == T{0}) {
    return;
  }
  for (int i = 0; i < m; ++i) {
    T w = c[i]; // column 0, v[0] == 1
    for (int j = 1; j < len; ++j) {
      w += c[i + static_cast<std::ptrdiff_t>(j) * ldc] * v[j];
    }
    w *= tau;
    c[i] -= w;
    for (int j = 1; j < len; ++j) {
      c[i + static_cast<std::ptrdiff_t>(j) * ldc] -= w * v[j];
    }
  }
}

// ==================================================================== DLAHQR
//
// Double-shift Francis QR for the Schur form of the whole JW-by-JW window T
// (always WANTT = WANTZ = .true., ILO = 1, IHI = JW here, accumulating into V),
// reference ?lahqr. On return the window of T is quasi-triangular, (wr, wi) hold
// the eigenvalues, V holds the orthogonal transform, and *infqr is 0 or the row
// that failed to converge. The body is calaman.lahqr's inlined kernel restricted
// to that call shape.
template<typename T>
__device__ void dlahqr_window(const int jw, T *t, const int ldt, T *wr, T *wi, T *v, const int ldv,
                              int *infqr) {
  const int n = jw;
  const int ilo = 1, ihi = jw;
  const int iloz = 1, ihiz = jw;
  auto Hidx = [=](int i, int j) -> std::ptrdiff_t {
    return (i - 1) + static_cast<std::ptrdiff_t>(j - 1) * ldt;
  };
  auto Vidx = [=](int i, int j) -> std::ptrdiff_t {
    return (i - 1) + static_cast<std::ptrdiff_t>(j - 1) * ldv;
  };
  auto H = [=](int i, int j) -> T & { return t[Hidx(i, j)]; };

  const T zero{0}, one{1}, two{2};
  *infqr = 0;
  if (n == 0) {
    return;
  }
  if (ilo == ihi) {
    wr[ilo - 1] = H(ilo, ilo);
    wi[ilo - 1] = zero;
    return;
  }
  for (int j = ilo; j <= ihi - 3; ++j) {
    H(j + 2, j) = zero;
    H(j + 3, j) = zero;
  }
  if (ilo <= ihi - 2) {
    H(ihi, ihi - 2) = zero;
  }
  const int nh = ihi - ilo + 1;
  const int nz = ihiz - iloz + 1;
  const T safmin = lamch<T>::sfmin();
  const T ulp = lamch<T>::eps();
  const T smlnum = safmin * (static_cast<T>(nh) / ulp);
  const T dat1 = T{3} / T{4};
  const T dat2 = T{-0.4375};
  int i1 = 1, i2 = n;
  const int itmax = 30 * (nh > 10 ? nh : 10);
  int kdefl = 0;
  T vv[3];
  int i = ihi;
  while (true) {
    int l = ilo;
    if (i < ilo) {
      break;
    }
    bool converged = false;
    for (int its = 0; its <= itmax; ++its) {
      int k;
      for (k = i; k >= l + 1; --k) {
        if (dabs(H(k, k - 1)) <= smlnum) {
          break;
        }
        T tst = dabs(H(k - 1, k - 1)) + dabs(H(k, k));
        if (tst == zero) {
          if (k - 2 >= ilo) {
            tst = tst + dabs(H(k - 1, k - 2));
          }
          if (k + 1 <= ihi) {
            tst = tst + dabs(H(k + 1, k));
          }
        }
        if (dabs(H(k, k - 1)) <= ulp * tst) {
          const T ab = dmax(dabs(H(k, k - 1)), dabs(H(k - 1, k)));
          const T ba = dmin(dabs(H(k, k - 1)), dabs(H(k - 1, k)));
          const T aa = dmax(dabs(H(k, k)), dabs(H(k - 1, k - 1) - H(k, k)));
          const T bb = dmin(dabs(H(k, k)), dabs(H(k - 1, k - 1) - H(k, k)));
          const T s = aa + ab;
          if (ba * (ab / s) <= dmax(smlnum, ulp * (bb * (aa / s)))) {
            break;
          }
        }
      }
      if (k < l + 1) {
        k = l;
      }
      l = k;
      if (l > ilo) {
        H(l, l - 1) = zero;
      }
      if (l >= i - 1) {
        converged = true;
        break;
      }
      kdefl = kdefl + 1;
      // WANTT is true here, so I1/I2 stay 1/n (never reset to the active window).
      T h11, h21, h12, h22;
      if (kdefl % (2 * 10) == 0) {
        const T s = dabs(H(i, i - 1)) + dabs(H(i - 1, i - 2));
        h11 = dat1 * s + H(i, i);
        h12 = dat2 * s;
        h21 = s;
        h22 = h11;
      } else if (kdefl % 10 == 0) {
        const T s = dabs(H(l + 1, l)) + dabs(H(l + 2, l + 1));
        h11 = dat1 * s + H(l, l);
        h12 = dat2 * s;
        h21 = s;
        h22 = h11;
      } else {
        h11 = H(i - 1, i - 1);
        h21 = H(i, i - 1);
        h12 = H(i - 1, i);
        h22 = H(i, i);
      }
      T rt1r, rt1i, rt2r, rt2i;
      {
        T s = dabs(h11) + dabs(h12) + dabs(h21) + dabs(h22);
        if (s == zero) {
          rt1r = zero;
          rt1i = zero;
          rt2r = zero;
          rt2i = zero;
        } else {
          h11 = h11 / s;
          h21 = h21 / s;
          h12 = h12 / s;
          h22 = h22 / s;
          const T tr = (h11 + h22) / two;
          const T det = (h11 - tr) * (h22 - tr) - h12 * h21;
          const T rtdisc = wwr::sqrt(dabs(det));
          if (det >= zero) {
            rt1r = tr * s;
            rt2r = rt1r;
            rt1i = rtdisc * s;
            rt2i = -rt1i;
          } else {
            rt1r = tr + rtdisc;
            rt2r = tr - rtdisc;
            if (dabs(rt1r - h22) <= dabs(rt2r - h22)) {
              rt1r = rt1r * s;
              rt2r = rt1r;
            } else {
              rt2r = rt2r * s;
              rt1r = rt2r;
            }
            rt1i = zero;
            rt2i = zero;
          }
        }
      }
      int m;
      for (m = i - 2; m >= l; --m) {
        T h21s = H(m + 1, m);
        T s = dabs(H(m, m) - rt2r) + dabs(rt2i) + dabs(h21s);
        h21s = H(m + 1, m) / s;
        vv[0] = h21s * H(m, m + 1) + (H(m, m) - rt1r) * ((H(m, m) - rt2r) / s) - rt1i * (rt2i / s);
        vv[1] = h21s * (H(m, m) + H(m + 1, m + 1) - rt1r - rt2r);
        vv[2] = h21s * H(m + 2, m + 1);
        s = dabs(vv[0]) + dabs(vv[1]) + dabs(vv[2]);
        vv[0] = vv[0] / s;
        vv[1] = vv[1] / s;
        vv[2] = vv[2] / s;
        if (m == l) {
          break;
        }
        if (dabs(H(m, m - 1)) * (dabs(vv[1]) + dabs(vv[2])) <=
            ulp * dabs(vv[0]) *
                (dabs(H(m - 1, m - 1)) + dabs(H(m, m)) + dabs(H(m + 1, m + 1)))) {
          break;
        }
      }
      if (m < l) {
        m = l;
      }
      for (int kk = m; kk <= i - 1; ++kk) {
        const int nr = dmin(3, i - kk + 1);
        if (kk > m) {
          for (int tt = 0; tt < nr; ++tt) {
            vv[tt] = H(kk + tt, kk - 1);
          }
        }
        T alpha = vv[0];
        const T t1 = dlarfg(nr, alpha, &vv[1], 1);
        vv[0] = alpha;
        if (kk > m) {
          H(kk, kk - 1) = vv[0];
          H(kk + 1, kk - 1) = zero;
          if (kk < i - 1) {
            H(kk + 2, kk - 1) = zero;
          }
        } else if (m > l) {
          H(kk, kk - 1) = H(kk, kk - 1) * (one - t1);
        }
        const T v2 = vv[1];
        const T t2 = t1 * v2;
        if (nr == 3) {
          const T v3 = vv[2];
          const T t3 = t1 * v3;
          for (int j = kk; j <= i2; ++j) {
            const T sum = H(kk, j) + v2 * H(kk + 1, j) + v3 * H(kk + 2, j);
            H(kk, j) = H(kk, j) - sum * t1;
            H(kk + 1, j) = H(kk + 1, j) - sum * t2;
            H(kk + 2, j) = H(kk + 2, j) - sum * t3;
          }
          const int jhi = dmin(kk + 3, i);
          for (int j = i1; j <= jhi; ++j) {
            const T sum = H(j, kk) + v2 * H(j, kk + 1) + v3 * H(j, kk + 2);
            H(j, kk) = H(j, kk) - sum * t1;
            H(j, kk + 1) = H(j, kk + 1) - sum * t2;
            H(j, kk + 2) = H(j, kk + 2) - sum * t3;
          }
          for (int j = iloz; j <= ihiz; ++j) {
            const T sum = v[Vidx(j, kk)] + v2 * v[Vidx(j, kk + 1)] + v3 * v[Vidx(j, kk + 2)];
            v[Vidx(j, kk)] = v[Vidx(j, kk)] - sum * t1;
            v[Vidx(j, kk + 1)] = v[Vidx(j, kk + 1)] - sum * t2;
            v[Vidx(j, kk + 2)] = v[Vidx(j, kk + 2)] - sum * t3;
          }
        } else if (nr == 2) {
          for (int j = kk; j <= i2; ++j) {
            const T sum = H(kk, j) + v2 * H(kk + 1, j);
            H(kk, j) = H(kk, j) - sum * t1;
            H(kk + 1, j) = H(kk + 1, j) - sum * t2;
          }
          for (int j = i1; j <= i; ++j) {
            const T sum = H(j, kk) + v2 * H(j, kk + 1);
            H(j, kk) = H(j, kk) - sum * t1;
            H(j, kk + 1) = H(j, kk + 1) - sum * t2;
          }
          for (int j = iloz; j <= ihiz; ++j) {
            const T sum = v[Vidx(j, kk)] + v2 * v[Vidx(j, kk + 1)];
            v[Vidx(j, kk)] = v[Vidx(j, kk)] - sum * t1;
            v[Vidx(j, kk + 1)] = v[Vidx(j, kk + 1)] - sum * t2;
          }
        }
      }
    }
    if (!converged) {
      *infqr = i;
      return;
    }
    if (l == i) {
      wr[i - 1] = H(i, i);
      wi[i - 1] = zero;
    } else if (l == i - 1) {
      T a = H(i - 1, i - 1), b = H(i - 1, i), c = H(i, i - 1), d = H(i, i);
      T r1r, r1i, r2r, r2i, cs, sn;
      dlanv2(a, b, c, d, r1r, r1i, r2r, r2i, cs, sn);
      H(i - 1, i - 1) = a;
      H(i - 1, i) = b;
      H(i, i - 1) = c;
      H(i, i) = d;
      wr[i - 2] = r1r;
      wi[i - 2] = r1i;
      wr[i - 1] = r2r;
      wi[i - 1] = r2i;
      if (i2 > i) {
        drot(i2 - i, &t[Hidx(i - 1, i + 1)], ldt, &t[Hidx(i, i + 1)], ldt, cs, sn);
      }
      drot(i - i1 - 1, &t[Hidx(i1, i - 1)], 1, &t[Hidx(i1, i)], 1, cs, sn);
      drot(nz, &v[Vidx(iloz, i - 1)], 1, &v[Vidx(iloz, i)], 1, cs, sn);
    }
    kdefl = 0;
    i = l - 1;
  }
}

// ==================================================================== DLAEXC
//
// Swap the n1-by-n1 block at j1 (1-based) with the adjacent n2-by-n2 block of the
// JW-by-JW Schur form T, accumulating into V (WANTQ always true). Returns *info =
// 1 on a rejected swap. calaman.laexc's inlined kernel, here over the window.
// ?lasy2 (the small Sylvester solve) is inlined above the swap it serves.

// ?lasy2: op(TL)*X + ISGN*X*op(TR) = SCALE*B, LTRANL=LTRANR=false, ISGN=-1.
template<typename T>
__device__ void dlasy2(const int n1, const int n2, const T *tl, const int ldtl, const T *tr,
                       const int ldtr, const T *b, const int ldb, T &scale, T *x, const int ldx) {
  const T eps = lamch<T>::eps();
  const T smlnum = lamch<T>::sfmin() / eps;
  const T sgn = T{-1};
  const T one = T{1};
  const T two = T{2};
  const T half = T{0.5};
  const T eight = T{8};
  auto TL = [=](int i, int j) -> T { return tl[(i - 1) + (j - 1) * ldtl]; };
  auto TR = [=](int i, int j) -> T { return tr[(i - 1) + (j - 1) * ldtr]; };
  auto B = [=](int i, int j) -> T { return b[(i - 1) + (j - 1) * ldb]; };
  auto setX = [=](int i, int j, T vv) { x[(i - 1) + (j - 1) * ldx] = vv; };
  const int kk = n1 + n1 + n2 - 2;
  if (kk == 1) {
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
    setX(1, 1, (B(1, 1) * sc) / tau1);
    scale = sc;
    return;
  }
  if (kk == 2 || kk == 3) {
    T tmp[5];
    T btmp[5];
    T smin;
    if (kk == 2) {
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
    const int kidx = 5 - i;
    const T temp = one / t16[kidx][kidx];
    tmp[kidx] = btmp[kidx] * temp;
    for (int j = kidx + 1; j <= 4; ++j) {
      tmp[kidx] = tmp[kidx] - (temp * t16[kidx][j]) * tmp[j];
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

// Order-3 ?larfg used by ?laexc's block-swap reflectors.
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

// Length-3 reflector apply (v stored explicitly, v[0] not implied 1), ?larfx form.
template<typename T>
__device__ void dlarfx3_left(const int m, const int n, const T *v, const T tau, T *c,
                             const int ldc) {
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

template<typename T>
__device__ void dlaexc_window(const int n, T *t, const int ldt, T *q, const int ldq, const int j1,
                              const int n1, const int n2, int *info) {
  auto Tidx = [=](int i, int j) -> std::ptrdiff_t {
    return (i - 1) + static_cast<std::ptrdiff_t>(j - 1) * ldt;
  };
  auto Qidx = [=](int i, int j) -> std::ptrdiff_t {
    return (i - 1) + static_cast<std::ptrdiff_t>(j - 1) * ldq;
  };
  auto Tref = [=](int i, int j) -> T & { return t[Tidx(i, j)]; };
  *info = 0;
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
    drot(n, &q[Qidx(1, j1)], 1, &q[Qidx(1, j2)], 1, cs, sn);
    return;
  }
  const int ldd = 4;
  const int ldx = 2;
  const int nd = n1 + n2;
  T d[16];
  for (int jj = 0; jj < nd; ++jj) {
    for (int ii = 0; ii < nd; ++ii) {
      d[ii + jj * ldd] = Tref(j1 + ii, j1 + jj);
    }
  }
  T dnorm = T{0};
  for (int jj = 0; jj < nd; ++jj) {
    for (int ii = 0; ii < nd; ++ii) {
      dnorm = dmax(dnorm, dabs(d[ii + jj * ldd]));
    }
  }
  const T eps = lamch<T>::eps();
  const T smlnum = lamch<T>::sfmin() / eps;
  const T thresh = dmax(T{10} * eps * dnorm, smlnum);
  T scale;
  T x[4];
  dlasy2(n1, n2, &d[0], ldd, &d[(n1) + (n1) * ldd], ldd, &d[0 + (n1) * ldd], ldd, scale, x, ldx);
  auto X = [&](int i, int j) -> T { return x[(i - 1) + (j - 1) * ldx]; };
  const int kk = n1 + n1 + n2 - 3;
  T u1[3], u2[3], u[3];
  T tau, tau1, tau2;
  if (kk == 1) {
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
      *info = 1;
      return;
    }
    dlarfx3_left(3, n - j1 + 1, u, tau, &t[Tidx(j1, j1)], ldt);
    dlarfx3_right(j2, 3, u, tau, &t[Tidx(1, j1)], ldt);
    Tref(j3, j1) = T{0};
    Tref(j3, j2) = T{0};
    Tref(j3, j3) = t11;
    dlarfx3_right(n, 3, u, tau, &q[Qidx(1, j1)], ldq);
  } else if (kk == 2) {
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
      *info = 1;
      return;
    }
    dlarfx3_right(j3, 3, u, tau, &t[Tidx(1, j1)], ldt);
    dlarfx3_left(3, n - j1, u, tau, &t[Tidx(j1, j2)], ldt);
    Tref(j1, j1) = t33;
    Tref(j2, j1) = T{0};
    Tref(j3, j1) = T{0};
    dlarfx3_right(n, 3, u, tau, &q[Qidx(1, j1)], ldq);
  } else {
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
    dlarfx3_left(3, 4, u2, tau2, &d[1], ldd);
    dlarfx3_right(4, 3, u2, tau2, &d[0 + 1 * ldd], ldd);
    if (dmax(dmax(dabs(d[2 + 0 * ldd]), dabs(d[2 + 1 * ldd])),
             dmax(dabs(d[3 + 0 * ldd]), dabs(d[3 + 1 * ldd]))) > thresh) {
      *info = 1;
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
    dlarfx3_right(n, 3, u1, tau1, &q[Qidx(1, j1)], ldq);
    dlarfx3_right(n, 3, u2, tau2, &q[Qidx(1, j2)], ldq);
  }
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
    drot(n, &q[Qidx(1, j1)], 1, &q[Qidx(1, j2)], 1, cs, sn);
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
    drot(n, &q[Qidx(1, jj3)], 1, &q[Qidx(1, jj4)], 1, cs, sn);
  }
}

// ==================================================================== DTREXC
//
// Move the block whose first row is *ifst to position *ilst in the JW-by-JW Schur
// form T, accumulating into V (COMPQ = 'V'). *info = 1 on a rejected swap.
// calaman.trexc's host bubbling loop, here over device memory with the swap being
// dlaexc_window. Reads T subdiagonals directly (no device round-trip).
template<typename T>
__device__ void dtrexc_window(const int n, T *t, const int ldt, T *q, const int ldq, int *ifst,
                             int *ilst, int *info) {
  auto Tref = [=](int i, int j) -> T & {
    return t[(i - 1) + static_cast<std::ptrdiff_t>(j - 1) * ldt];
  };
  *info = 0;
  if (n <= 1) {
    return;
  }
  int ifst_ = *ifst;
  int ilst_ = *ilst;
  if (ifst_ > 1) {
    if (Tref(ifst_, ifst_ - 1) != T{0}) {
      --ifst_;
    }
  }
  int nbf = 1;
  if (ifst_ < n) {
    if (Tref(ifst_ + 1, ifst_) != T{0}) {
      nbf = 2;
    }
  }
  if (ilst_ > 1) {
    if (Tref(ilst_, ilst_ - 1) != T{0}) {
      --ilst_;
    }
  }
  int nbl = 1;
  if (ilst_ < n) {
    if (Tref(ilst_ + 1, ilst_) != T{0}) {
      nbl = 2;
    }
  }
  if (ifst_ == ilst_) {
    *ifst = ifst_;
    *ilst = ilst_;
    return;
  }
  int here = ifst_;
  if (ifst_ < ilst_) {
    if (nbf == 2 && nbl == 1) {
      --ilst_;
    }
    if (nbf == 1 && nbl == 2) {
      ++ilst_;
    }
    do {
      if (nbf == 1 || nbf == 2) {
        int nbnext = 1;
        if (here + nbf + 1 <= n && Tref(here + nbf + 1, here + nbf) != T{0}) {
          nbnext = 2;
        }
        int hinfo = 0;
        dlaexc_window(n, t, ldt, q, ldq, here, nbf, nbnext, &hinfo);
        if (hinfo != 0) {
          *info = 1;
          break;
        }
        here += nbnext;
        if (nbf == 2 && Tref(here + 1, here) == T{0}) {
          nbf = 3;
        }
      } else {
        int nbnext = 1;
        if (here + 3 <= n && Tref(here + 3, here + 2) != T{0}) {
          nbnext = 2;
        }
        int hinfo = 0;
        dlaexc_window(n, t, ldt, q, ldq, here + 1, 1, nbnext, &hinfo);
        if (hinfo != 0) {
          *info = 1;
          break;
        }
        if (nbnext == 1) {
          dlaexc_window(n, t, ldt, q, ldq, here, 1, nbnext, &hinfo);
          ++here;
        } else {
          if (Tref(here + 2, here + 1) == T{0}) {
            nbnext = 1;
          }
          if (nbnext == 2) {
            dlaexc_window(n, t, ldt, q, ldq, here, 1, nbnext, &hinfo);
            if (hinfo != 0) {
              *info = 1;
              break;
            }
            here += 2;
          } else {
            dlaexc_window(n, t, ldt, q, ldq, here, 1, 1, &hinfo);
            dlaexc_window(n, t, ldt, q, ldq, here + 1, 1, 1, &hinfo);
            here += 2;
          }
        }
      }
    } while (here < ilst_);
  } else {
    do {
      if (nbf == 1 || nbf == 2) {
        int nbnext = 1;
        if (here >= 3 && Tref(here - 1, here - 2) != T{0}) {
          nbnext = 2;
        }
        int hinfo = 0;
        dlaexc_window(n, t, ldt, q, ldq, here - nbnext, nbnext, nbf, &hinfo);
        if (hinfo != 0) {
          *info = 1;
          break;
        }
        here -= nbnext;
        if (nbf == 2 && Tref(here + 1, here) == T{0}) {
          nbf = 3;
        }
      } else {
        int nbnext = 1;
        if (here >= 3 && Tref(here - 1, here - 2) != T{0}) {
          nbnext = 2;
        }
        int hinfo = 0;
        dlaexc_window(n, t, ldt, q, ldq, here - nbnext, nbnext, 1, &hinfo);
        if (hinfo != 0) {
          *info = 1;
          break;
        }
        if (nbnext == 1) {
          dlaexc_window(n, t, ldt, q, ldq, here, nbnext, 1, &hinfo);
          --here;
        } else {
          if (Tref(here, here - 1) == T{0}) {
            nbnext = 1;
          }
          if (nbnext == 2) {
            dlaexc_window(n, t, ldt, q, ldq, here - 1, 2, 1, &hinfo);
            if (hinfo != 0) {
              *info = 1;
              break;
            }
            here -= 2;
          } else {
            dlaexc_window(n, t, ldt, q, ldq, here, 1, 1, &hinfo);
            dlaexc_window(n, t, ldt, q, ldq, here - 1, 1, 1, &hinfo);
            here -= 2;
          }
        }
      }
    } while (here > ilst_);
  }
  ilst_ = here;
  *ifst = ifst_;
  *ilst = ilst_;
}

// ==================================================================== DGEHRD
//
// Unblocked reduction of T over the window [ilo=1, ihi] to upper Hessenberg form
// (reference ?gehd2 with n = jw, ihi = ns): reflector i zeros T(i+2:ihi, i),
// stored with its implicit unit head at T(i+1, i) and tail below, scalar in
// tau[i-1]. The AED call reduces columns 1..ns, so ihi = ns here.
template<typename T>
__device__ void dgehrd_window(const int jw, const int ihi, T *t, const int ldt, T *tau) {
  for (int i = 1; i <= ihi - 1; ++i) {
    const int order = ihi - i; // reflector order = ihi - i
    // v at T(i+1, i); dlarfg overwrites T(i+1,i)=beta, tail T(i+2:ihi,i), and the
    // unit head is implicit for the applies below (v[0] treated as 1).
    T *v = &t[(i + 1 - 1) + static_cast<std::ptrdiff_t>(i - 1) * ldt];
    T alpha = *v;
    const T t1 = dlarfg(order, alpha, v + 1, 1);
    tau[i - 1] = t1;
    const T beta = alpha;
    *v = T{1};
    // Apply H(i) from the right to T(1:ihi, i+1:ihi): m = ihi, n = order.
    dlarf_right(ihi, order, v, t1, &t[0 + static_cast<std::ptrdiff_t>(i) * ldt], ldt);
    // Apply H(i) from the left to T(i+1:ihi, i+1:jw): m = order, n = jw - i.
    dlarf_left(order, jw - i, v, t1, &t[(i + 1 - 1) + static_cast<std::ptrdiff_t>(i) * ldt], ldt);
    *v = beta;
  }
}

// ==================================================================== DORMHR
//
// Apply the orthogonal Q from dgehrd_window to V from the right, V <- V*Q, the
// reference's DORMHR('R','N', jw, ns, 1, ns, T, WORK(tau), V): Q = H(1) H(2) ...
// H(ns-1). For C*Q with H(1) first, ?ormqr('R','N') applies the reflectors in
// forward order; reflector i has unit head at T(i+1, i) and tail at T(i+2:ns, i),
// and multiplies V's columns i+1..ns over all jw rows.
template<typename T>
__device__ void dormhr_window(const int jw, const int ns, T *t, const int ldt, const T *tau, T *v,
                             const int ldv) {
  for (int i = 1; i <= ns - 1; ++i) {
    const int order = ns - i; // reflector length (rows i+1..ns)
    T *vv = &t[(i + 1 - 1) + static_cast<std::ptrdiff_t>(i - 1) * ldt];
    const T beta = *vv;
    *vv = T{1}; // unit head for the apply
    // Right apply to V(1:jw, i+1:i+order).
    dlarf_right(jw, order, vv, tau[i - 1], &v[0 + static_cast<std::ptrdiff_t>(i) * ldv], ldv);
    *vv = beta;
  }
}

// The whole AED pass, run by one thread.
template<typename T>
struct Laqr2Functor {
  const bool wantt_;
  const bool wantz_;
  const int n_;
  const int ktop_;
  const int kbot_;
  const int nw_;
  T *const h_;
  const int ldh_;
  const int iloz_;
  const int ihiz_;
  T *const z_;
  const int ldz_;
  int *const ns_;
  int *const nd_;
  T *const sr_;
  T *const si_;
  T *const v_;
  const int ldv_;
  T *const t_;
  const int ldt_;
  T *const wv_; // slab scratch, unused (unblocked)
  const int ldwv_;

  __device__ void operator()(std::size_t) const {
    const T zero{0}, one{1};
    const int n = n_;
    const int ktop = ktop_;
    const int kbot = kbot_;
    const int ldh = ldh_;
    const int ldv = ldv_;
    const int ldt = ldt_;
    const int ldz = ldz_;
    const bool wantt = wantt_;
    const bool wantz = wantz_;
    const int iloz = iloz_;
    const int ihiz = ihiz_;
    T *const h = h_;
    T *const z = z_;
    T *const v = v_;
    T *const tt = t_;
    T *const sr = sr_;
    T *const si = si_;

    auto H = [=](int i, int j) -> T & {
      return h[(i - 1) + static_cast<std::ptrdiff_t>(j - 1) * ldh];
    };
    auto Z = [=](int i, int j) -> T & {
      return z[(i - 1) + static_cast<std::ptrdiff_t>(j - 1) * ldz];
    };
    auto V = [=](int i, int j) -> T & {
      return v[(i - 1) + static_cast<std::ptrdiff_t>(j - 1) * ldv];
    };
    auto Tw = [=](int i, int j) -> T & {
      return tt[(i - 1) + static_cast<std::ptrdiff_t>(j - 1) * ldt];
    };

    *ns_ = 0;
    *nd_ = 0;
    if (ktop > kbot) {
      return;
    }
    if (nw_ < 1) {
      return;
    }

    const T safmin = lamch<T>::sfmin();
    const T ulp = lamch<T>::eps();
    const T smlnum = safmin * (static_cast<T>(n) / ulp);

    int jw = dmin(nw_, kbot - ktop + 1);
    const int kwtop = kbot - jw + 1;
    T s;
    if (kwtop == ktop) {
      s = zero;
    } else {
      s = H(kwtop, kwtop - 1);
    }

    if (kbot == kwtop) {
      // 1-by-1 deflation window.
      sr[kwtop - 1] = H(kwtop, kwtop);
      si[kwtop - 1] = zero;
      int ns = 1, nd = 0;
      if (dabs(s) <= dmax(smlnum, ulp * dabs(H(kwtop, kwtop)))) {
        ns = 0;
        nd = 1;
        if (kwtop > ktop) {
          H(kwtop, kwtop - 1) = zero;
        }
      }
      *ns_ = ns;
      *nd_ = nd;
      return;
    }

    // Convert to spike-triangular form: copy the upper triangle and subdiagonal
    // of H's window into T, V := I, then ?lahqr the window.
    for (int j = 1; j <= jw; ++j) {
      for (int i = 1; i <= j; ++i) {
        Tw(i, j) = H(kwtop + i - 1, kwtop + j - 1);
      }
    }
    for (int i = 1; i <= jw - 1; ++i) {
      Tw(i + 1, i) = H(kwtop + i, kwtop + i - 1);
    }
    for (int j = 1; j <= jw; ++j) {
      for (int i = 1; i <= jw; ++i) {
        V(i, j) = (i == j) ? one : zero;
      }
    }
    int infqr = 0;
    dlahqr_window(jw, tt, ldt, &sr[kwtop - 1], &si[kwtop - 1], v, ldv, &infqr);

    // DTREXC needs a clean margin near the diagonal.
    for (int j = 1; j <= jw - 3; ++j) {
      Tw(j + 2, j) = zero;
      Tw(j + 3, j) = zero;
    }
    if (jw > 2) {
      Tw(jw, jw - 2) = zero;
    }

    // Deflation detection loop.
    int ns = jw;
    int ilst = infqr + 1;
    while (ilst <= ns) {
      bool bulge;
      if (ns == 1) {
        bulge = false;
      } else {
        bulge = Tw(ns, ns - 1) != zero;
      }
      if (!bulge) {
        T foo = dabs(Tw(ns, ns));
        if (foo == zero) {
          foo = dabs(s);
        }
        if (dabs(s * V(1, ns)) <= dmax(smlnum, ulp * foo)) {
          ns = ns - 1;
        } else {
          int ifst = ns;
          int info = 0;
          dtrexc_window(jw, tt, ldt, v, ldv, &ifst, &ilst, &info);
          ilst = ilst + 1;
        }
      } else {
        T foo = dabs(Tw(ns, ns)) +
                wwr::sqrt(dabs(Tw(ns, ns - 1))) * wwr::sqrt(dabs(Tw(ns - 1, ns)));
        if (foo == zero) {
          foo = dabs(s);
        }
        if (dmax(dabs(s * V(1, ns)), dabs(s * V(1, ns - 1))) <= dmax(smlnum, ulp * foo)) {
          ns = ns - 2;
        } else {
          int ifst = ns;
          int info = 0;
          dtrexc_window(jw, tt, ldt, v, ldv, &ifst, &ilst, &info);
          ilst = ilst + 2;
        }
      }
    }

    if (ns == 0) {
      s = zero;
    }

    if (ns < jw) {
      // Bubble sort the remaining diagonal blocks by eigenvalue magnitude.
      bool sorted = false;
      int i = ns + 1;
      while (!sorted) {
        sorted = true;
        const int kend = i - 1;
        i = infqr + 1;
        int k;
        if (i == ns) {
          k = i + 1;
        } else if (Tw(i + 1, i) == zero) {
          k = i + 1;
        } else {
          k = i + 2;
        }
        while (k <= kend) {
          T evi;
          if (k == i + 1) {
            evi = dabs(Tw(i, i));
          } else {
            evi = dabs(Tw(i, i)) + wwr::sqrt(dabs(Tw(i + 1, i))) * wwr::sqrt(dabs(Tw(i, i + 1)));
          }
          T evk;
          if (k == kend) {
            evk = dabs(Tw(k, k));
          } else if (Tw(k + 1, k) == zero) {
            evk = dabs(Tw(k, k));
          } else {
            evk = dabs(Tw(k, k)) + wwr::sqrt(dabs(Tw(k + 1, k))) * wwr::sqrt(dabs(Tw(k, k + 1)));
          }
          if (evi >= evk) {
            i = k;
          } else {
            sorted = false;
            int ifst = i;
            int ilst2 = k;
            int info = 0;
            dtrexc_window(jw, tt, ldt, v, ldv, &ifst, &ilst2, &info);
            if (info == 0) {
              i = ilst2;
            } else {
              i = k;
            }
          }
          if (i == kend) {
            k = i + 1;
          } else if (Tw(i + 1, i) == zero) {
            k = i + 1;
          } else {
            k = i + 2;
          }
        }
      }
    }

    // Restore shift/eigenvalue array from T.
    {
      int i = jw;
      while (i >= infqr + 1) {
        if (i == infqr + 1) {
          sr[kwtop + i - 2] = Tw(i, i);
          si[kwtop + i - 2] = zero;
          i = i - 1;
        } else if (Tw(i, i - 1) == zero) {
          sr[kwtop + i - 2] = Tw(i, i);
          si[kwtop + i - 2] = zero;
          i = i - 1;
        } else {
          T aa = Tw(i - 1, i - 1);
          T cc = Tw(i, i - 1);
          T bb = Tw(i - 1, i);
          T dd = Tw(i, i);
          T cs, sn;
          dlanv2(aa, bb, cc, dd, sr[kwtop + i - 3], si[kwtop + i - 3], sr[kwtop + i - 2],
                 si[kwtop + i - 2], cs, sn);
          i = i - 2;
        }
      }
    }

    if (ns < jw || s == zero) {
      if (ns > 1 && s != zero) {
        // Reflect spike back into the lower triangle: the reference's DCOPY(NS,
        // V, LDV, WORK, 1) takes the first ROW V(1, 1:ns) (stride LDV), not the
        // first column; ?larfg it, then apply to T (both sides) and V (right).
        T wrk[kMaxWindow];
        for (int ii = 0; ii < ns; ++ii) {
          wrk[ii] = V(1, 1 + ii);
        }
        T beta = wrk[0];
        const T tau = dlarfg(ns, beta, &wrk[1], 1);
        wrk[0] = one;
        // DLASET 'L' of T(3:jw, 1:jw-2) to zero.
        for (int j = 1; j <= jw - 2; ++j) {
          for (int i = 3 + (j - 1); i <= jw; ++i) {
            Tw(i, j) = zero;
          }
        }
        // T <- H_v T (left, length ns over jw columns).
        dlarf_left(ns, jw, wrk, tau, &tt[0], ldt);
        // T <- T H_v (right, length ns over ns rows).
        dlarf_right(ns, ns, wrk, tau, &tt[0], ldt);
        // V <- V H_v (right, length ns over jw rows).
        dlarf_right(jw, ns, wrk, tau, &v[0], ldv);
        // Re-Hessenberg columns 1..ns of T.
        T tauh[kMaxWindow];
        dgehrd_window(jw, ns, tt, ldt, tauh);
        // Accumulate the gehrd Q into V from the right.
        dormhr_window(jw, ns, tt, ldt, tauh, v, ldv);
      }

      // Copy the updated reduced window back into H.
      if (kwtop > 1) {
        H(kwtop, kwtop - 1) = s * V(1, 1);
      }
      for (int j = 1; j <= jw; ++j) {
        for (int i = 1; i <= j; ++i) {
          H(kwtop + i - 1, kwtop + j - 1) = Tw(i, j);
        }
      }
      for (int i = 1; i <= jw - 1; ++i) {
        H(kwtop + i, kwtop + i - 1) = Tw(i + 1, i);
      }

      // Update vertical slab in H: H(ltop:kwtop-1, kwtop:kwtop+jw-1) *= V.
      const int ltop = wantt ? 1 : ktop;
      for (int row = ltop; row <= kwtop - 1; ++row) {
        T acc[kMaxWindow];
        for (int c = 1; c <= jw; ++c) {
          T sum = zero;
          for (int kk = 1; kk <= jw; ++kk) {
            sum += H(row, kwtop + kk - 1) * V(kk, c);
          }
          acc[c - 1] = sum;
        }
        for (int c = 1; c <= jw; ++c) {
          H(row, kwtop + c - 1) = acc[c - 1];
        }
      }

      // Update horizontal slab in H (WANTT): H(kwtop:.., kbot+1:n) = V^T * H(..).
      if (wantt) {
        for (int col = kbot + 1; col <= n; ++col) {
          T acc[kMaxWindow];
          for (int r = 1; r <= jw; ++r) {
            T sum = zero;
            for (int kk = 1; kk <= jw; ++kk) {
              sum += V(kk, r) * H(kwtop + kk - 1, col);
            }
            acc[r - 1] = sum;
          }
          for (int r = 1; r <= jw; ++r) {
            H(kwtop + r - 1, col) = acc[r - 1];
          }
        }
      }

      // Update vertical slab in Z: Z(iloz:ihiz, kwtop:..) *= V.
      if (wantz) {
        for (int row = iloz; row <= ihiz; ++row) {
          T acc[kMaxWindow];
          for (int c = 1; c <= jw; ++c) {
            T sum = zero;
            for (int kk = 1; kk <= jw; ++kk) {
              sum += Z(row, kwtop + kk - 1) * V(kk, c);
            }
            acc[c - 1] = sum;
          }
          for (int c = 1; c <= jw; ++c) {
            Z(row, kwtop + c - 1) = acc[c - 1];
          }
        }
      }
    }

    *nd_ = jw - ns;
    *ns_ = ns - infqr;
  }
};

} // namespace

template<typename T>
void laqr2(const wwr::wwrStream_t stream, const bool wantt, const bool wantz, const int n,
           const int ktop, const int kbot, const int nw, T *const h, const int ldh, const int iloz,
           const int ihiz, T *const z, const int ldz, int *const ns, int *const nd, T *const sr,
           T *const si, T *const v, const int ldv, const int nh, T *const t, const int ldt,
           const int nv, T *const wv, const int ldwv) {
  (void)nh;
  (void)nv;
  const Laqr2Functor<T> functor{wantt, wantz, n,  ktop, kbot, nw,  h,  ldh, iloz, ihiz, z, ldz,
                                 ns,    nd,    sr, si,   v,    ldv, t,  ldt, wv,   ldwv};
  wwr::extension::parallel_for<std::size_t>(stream, std::size_t{1}, functor);
}

// One per supported precision, matching laqr2_bridge.h's declaration and
// interface.cppm's extern-template list -- a type added here without being added
// there, or vice versa, links against nothing.
template void laqr2<float>(wwr::wwrStream_t, bool, bool, int, int, int, int, float *, int, int, int,
                           float *, int, int *, int *, float *, float *, float *, int, int, float *,
                           int, int, float *, int);
template void laqr2<double>(wwr::wwrStream_t, bool, bool, int, int, int, int, double *, int, int,
                            int, double *, int, int *, int *, double *, double *, double *, int, int,
                            double *, int, int, double *, int);

} // namespace calaman::device
