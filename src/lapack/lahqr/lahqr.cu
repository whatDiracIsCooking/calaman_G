// lahqr.cu
//
// The device-kernel half of calaman.lahqr: a single-thread port of LAPACK's
// ?lahqr, launched through wwr.extension.parallel_for over a grid of one. The
// routine computes the real Schur form of an upper Hessenberg matrix by the
// double-shift Francis QR iteration -- the QR-iteration base case the real
// nonsymmetric eigensolver (?hseqr / geev) recurses down to. One thread runs the
// entire reference algorithm: the subdiagonal deflation search, the Francis
// double/exceptional shift, the bulge-chasing sweep of length-3 reflectors, and
// the ?lanv2 standardisation of each converged 2x2 block, so there is no host
// round-trip and the module wrapper is a thin front door, the shape of
// calaman.laexc.
//
// FAITHFUL TO THE REFERENCE, by design: the test oracle is reference ?lahqr, so
// the structure mirrors dlahqr.f line for line -- the Ahues & Tisseur deflation
// criterion, the KEXSH=10 exceptional-shift cadence with DAT1=3/4 and
// DAT2=-0.4375, the scaled shift computation, the two-consecutive-subdiagonal
// start search, and the NR in {2,3} reflector applies over the [I1,I2] window.
// The auxiliaries it calls (DLARFG, DLANV2, DROT, DCOPY) are inlined here as
// __device__ helpers so the whole reduction is one kernel. DLAMCH('P')/('S') are
// the <cfloat> macros FLT/DBL_EPSILON and FLT/DBL_MIN -- usable on the device,
// where std::numeric_limits is not (its members are __host__ constexpr, rejected
// by nvcc). Indices track the Fortran 1-based scheme; the column-major (i,j)
// accessors fold the leading dimension out.
//
// REAL ONLY: LAPACK's double-shift QR is the real path (a complex Hessenberg
// uses the single-shift ?lahqr's complex sibling), so the surface is float /
// double, matching the reference. Shared unchanged between both backends: a .cu
// is compiled by the backend compiler, and parallel_for's launch machinery plus
// the precision-neutral math (sqrt/fabs/hypot/copysign) arrive through device
// headers.
#include "lahqr_bridge.h"

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

// abs / max / min in exact arithmetic, matching LAPACK's ABS / MAX / MIN bit for
// bit so the control flow (the deflation tests, the shift selection) matches.
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
// Apply the plane rotation (cs, sn) to the pair of vectors (x, y), each of n
// elements with the given stride: x,y <- (cs*x + sn*y, cs*y - sn*x). The Level-1
// BLAS DROT ?lahqr uses to mix the two rows/columns of a converged 2x2 block.
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

// ------------------------------------------------------------------ DLARFG
//
// Generate the order-n (n in {2,3}) Householder reflector H = I - tau*v*v^T with
// v=[1;v2;...] such that H*[alpha;x] = [beta;0], reference ?larfg. On entry
// (alpha, x[0..n-2]) are the n input scalars; on return alpha holds beta,
// x[0..n-2] hold the reflector tail, and tau is returned. The unit leading entry
// is implied (?lahqr keeps v(1) separately). n==1 => tau=0 (no reflector).
template<typename T>
__device__ void dlarfg(const int n, T &alpha, T *x, T &tau) {
  if (n <= 1) {
    tau = T{0};
    return;
  }
  // xnorm = ||x[0..n-2]||_2
  T xnorm = T{0};
  for (int i = 0; i < n - 1; ++i) {
    xnorm = wwr::hypot(xnorm, x[i]);
  }
  if (xnorm == T{0}) {
    tau = T{0};
    return;
  }
  const T r = wwr::hypot(alpha, xnorm);
  const T beta = alpha >= T{0} ? -r : r;
  tau = (beta - alpha) / beta;
  const T scal = T{1} / (alpha - beta);
  for (int i = 0; i < n - 1; ++i) {
    x[i] *= scal;
  }
  alpha = beta;
}

// ------------------------------------------------------------------ DLANV2
//
// Schur-standardise the real 2x2 block [a b; c d], reference ?lanv2 verbatim (V.
// Sima's cancellation-reducing revision). a, b, c, d are updated in place,
// (rt1r, rt1i) / (rt2r, rt2i) receive the two eigenvalues (rt1i > 0 for a complex
// pair), and (cs, sn) the rotation so the caller can rotate the rest of H / Z.
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

  // Store the eigenvalues.
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

// LAPACK DAT1 = 3/4, DAT2 = -0.4375 and the KEXSH = 10 exceptional-shift cadence.
template<typename T>
__device__ __forceinline__ T dat1() {
  return T{3} / T{4};
}
template<typename T>
__device__ __forceinline__ T dat2() {
  return T{-0.4375};
}

// The whole QR reduction, run by one thread. Members are const scalar / pointer,
// as device_functor requires: trivially copyable, and the const deletes the
// copy-assignment the grid-constant kernel copy would otherwise permit. The
// index is ignored (the grid is a single element).
template<typename T>
struct LahqrFunctor {
  const bool wantt_;
  const bool wantz_;
  const int n_;
  const int ilo_; // 1-based
  const int ihi_; // 1-based
  T *const h_;
  const int ldh_;
  T *const wr_;
  T *const wi_;
  const int iloz_; // 1-based
  const int ihiz_; // 1-based
  T *const z_;
  const int ldz_;
  int *const info_;

  __device__ void operator()(std::size_t) const {
    const int n = n_;
    const int ilo = ilo_;
    const int ihi = ihi_;
    const int ldh = ldh_;
    const int ldz = ldz_;
    const int iloz = iloz_;
    const int ihiz = ihiz_;
    T *const h = h_;
    T *const z = z_;
    T *const wr = wr_;
    T *const wi = wi_;

    // Column-major 1-based accessors into H (and Z).
    auto Hidx = [=](int i, int j) -> std::ptrdiff_t {
      return (i - 1) + static_cast<std::ptrdiff_t>(j - 1) * ldh;
    };
    auto Zidx = [=](int i, int j) -> std::ptrdiff_t {
      return (i - 1) + static_cast<std::ptrdiff_t>(j - 1) * ldz;
    };
    auto H = [=](int i, int j) -> T & { return h[Hidx(i, j)]; };

    const T zero{0}, one{1}, two{2};

    *info_ = 0;

    // Quick return.
    if (n == 0) {
      return;
    }
    if (ilo == ihi) {
      wr[ilo - 1] = H(ilo, ilo);
      wi[ilo - 1] = zero;
      return;
    }

    // ==== clear out the trash ====
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

    // I1 and I2: the first row and last column transformations touch. When only
    // eigenvalues are wanted they are reset per active window inside the loop.
    int i1 = 1, i2 = n;

    const int itmax = 30 * (nh > 10 ? nh : 10);
    int kdefl = 0;

    T v[3];

    // Main loop: I decreases from IHI to ILO in steps of 1 or 2.
    int i = ihi;
    while (true) {
      int l = ilo;
      if (i < ilo) {
        break;
      }

      bool converged = false;
      int its = 0;
      for (its = 0; its <= itmax; ++its) {
        // Look for a single small subdiagonal element.
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
          // Ahues & Tisseur deflation criterion.
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
          k = l; // DO loop fell through: K = L.
        }
        l = k;
        if (l > ilo) {
          H(l, l - 1) = zero; // H(L,L-1) is negligible.
        }

        // Exit if a submatrix of order 1 or 2 has split off.
        if (l >= i - 1) {
          converged = true;
          break;
        }
        kdefl = kdefl + 1;

        if (!wantt_) {
          i1 = l;
          i2 = i;
        }

        // Compute the shift (H11,H21,H12,H22), scaled, into the roots.
        T h11, h21, h12, h22;
        if (kdefl % (2 * 10) == 0) {
          // Exceptional shift (bottom).
          const T s = dabs(H(i, i - 1)) + dabs(H(i - 1, i - 2));
          h11 = dat1<T>() * s + H(i, i);
          h12 = dat2<T>() * s;
          h21 = s;
          h22 = h11;
        } else if (kdefl % 10 == 0) {
          // Exceptional shift (top).
          const T s = dabs(H(l + 1, l)) + dabs(H(l + 2, l + 1));
          h11 = dat1<T>() * s + H(l, l);
          h12 = dat2<T>() * s;
          h21 = s;
          h22 = h11;
        } else {
          // Francis double shift from the trailing 2x2.
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
              // complex conjugate shifts
              rt1r = tr * s;
              rt2r = rt1r;
              rt1i = rtdisc * s;
              rt2i = -rt1i;
            } else {
              // real shifts (use only one of them)
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

        // Look for two consecutive small subdiagonal elements.
        int m;
        for (m = i - 2; m >= l; --m) {
          T h21s = H(m + 1, m);
          T s = dabs(H(m, m) - rt2r) + dabs(rt2i) + dabs(h21s);
          h21s = H(m + 1, m) / s;
          v[0] = h21s * H(m, m + 1) +
                 (H(m, m) - rt1r) * ((H(m, m) - rt2r) / s) - rt1i * (rt2i / s);
          v[1] = h21s * (H(m, m) + H(m + 1, m + 1) - rt1r - rt2r);
          v[2] = h21s * H(m + 2, m + 1);
          s = dabs(v[0]) + dabs(v[1]) + dabs(v[2]);
          v[0] = v[0] / s;
          v[1] = v[1] / s;
          v[2] = v[2] / s;
          if (m == l) {
            break;
          }
          if (dabs(H(m, m - 1)) * (dabs(v[1]) + dabs(v[2])) <=
              ulp * dabs(v[0]) *
                  (dabs(H(m - 1, m - 1)) + dabs(H(m, m)) + dabs(H(m + 1, m + 1)))) {
            break;
          }
        }
        if (m < l) {
          m = l;
        }

        // Double-shift QR step: chase the bulge from M to I-1.
        for (int k = m; k <= i - 1; ++k) {
          const int nr = dmin(3, i - k + 1);
          if (k > m) {
            // DCOPY(NR, H(K,K-1), 1, V, 1)
            for (int t = 0; t < nr; ++t) {
              v[t] = H(k + t, k - 1);
            }
          }
          T t1;
          dlarfg(nr, v[0], &v[1], t1);
          if (k > m) {
            H(k, k - 1) = v[0];
            H(k + 1, k - 1) = zero;
            if (k < i - 1) {
              H(k + 2, k - 1) = zero;
            }
          } else if (m > l) {
            // Avoid a bug when v(2),v(3) underflow.
            H(k, k - 1) = H(k, k - 1) * (one - t1);
          }
          const T v2 = v[1];
          const T t2 = t1 * v2;
          if (nr == 3) {
            const T v3 = v[2];
            const T t3 = t1 * v3;
            // Apply G from the left, columns K to I2.
            for (int j = k; j <= i2; ++j) {
              const T sum = H(k, j) + v2 * H(k + 1, j) + v3 * H(k + 2, j);
              H(k, j) = H(k, j) - sum * t1;
              H(k + 1, j) = H(k + 1, j) - sum * t2;
              H(k + 2, j) = H(k + 2, j) - sum * t3;
            }
            // Apply G from the right, rows I1 to min(K+3,I).
            const int jhi = dmin(k + 3, i);
            for (int j = i1; j <= jhi; ++j) {
              const T sum = H(j, k) + v2 * H(j, k + 1) + v3 * H(j, k + 2);
              H(j, k) = H(j, k) - sum * t1;
              H(j, k + 1) = H(j, k + 1) - sum * t2;
              H(j, k + 2) = H(j, k + 2) - sum * t3;
            }
            if (wantz_) {
              for (int j = iloz; j <= ihiz; ++j) {
                const T sum =
                    z[Zidx(j, k)] + v2 * z[Zidx(j, k + 1)] + v3 * z[Zidx(j, k + 2)];
                z[Zidx(j, k)] = z[Zidx(j, k)] - sum * t1;
                z[Zidx(j, k + 1)] = z[Zidx(j, k + 1)] - sum * t2;
                z[Zidx(j, k + 2)] = z[Zidx(j, k + 2)] - sum * t3;
              }
            }
          } else if (nr == 2) {
            // Apply G from the left, columns K to I2.
            for (int j = k; j <= i2; ++j) {
              const T sum = H(k, j) + v2 * H(k + 1, j);
              H(k, j) = H(k, j) - sum * t1;
              H(k + 1, j) = H(k + 1, j) - sum * t2;
            }
            // Apply G from the right, rows I1 to I.
            for (int j = i1; j <= i; ++j) {
              const T sum = H(j, k) + v2 * H(j, k + 1);
              H(j, k) = H(j, k) - sum * t1;
              H(j, k + 1) = H(j, k + 1) - sum * t2;
            }
            if (wantz_) {
              for (int j = iloz; j <= ihiz; ++j) {
                const T sum = z[Zidx(j, k)] + v2 * z[Zidx(j, k + 1)];
                z[Zidx(j, k)] = z[Zidx(j, k)] - sum * t1;
                z[Zidx(j, k + 1)] = z[Zidx(j, k + 1)] - sum * t2;
              }
            }
          }
        }
      }

      if (!converged) {
        // Failure to converge in the remaining iterations.
        *info_ = i;
        return;
      }

      if (l == i) {
        // One eigenvalue has converged.
        wr[i - 1] = H(i, i);
        wi[i - 1] = zero;
      } else if (l == i - 1) {
        // A pair of eigenvalues has converged: standardise the 2x2.
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
        if (wantt_) {
          if (i2 > i) {
            drot(i2 - i, &h[Hidx(i - 1, i + 1)], ldh, &h[Hidx(i, i + 1)], ldh, cs, sn);
          }
          drot(i - i1 - 1, &h[Hidx(i1, i - 1)], 1, &h[Hidx(i1, i)], 1, cs, sn);
        }
        if (wantz_) {
          drot(nz, &z[Zidx(iloz, i - 1)], 1, &z[Zidx(iloz, i)], 1, cs, sn);
        }
      }

      kdefl = 0;
      i = l - 1; // return to the top with the new I
    }
  }
};

} // namespace

template<typename T>
void lahqr(const wwr::wwrStream_t stream, const bool wantt, const bool wantz, const int n,
           const int ilo, const int ihi, T *const h, const int ldh, T *const wr, T *const wi,
           const int iloz, const int ihiz, T *const z, const int ldz, int *const info) {
  const LahqrFunctor<T> functor{wantt, wantz, n,    ilo,  ihi,  h, ldh,
                                 wr,    wi,    iloz, ihiz, z,    ldz, info};
  wwr::extension::parallel_for<std::size_t>(stream, std::size_t{1}, functor);
}

// One per supported precision, matching lahqr_bridge.h's declaration and
// interface.cppm's extern-template list -- a type added here without being added
// there, or vice versa, links against nothing.
template void lahqr<float>(wwr::wwrStream_t, bool, bool, int, int, int, float *, int, float *,
                           float *, int, int, float *, int, int *);
template void lahqr<double>(wwr::wwrStream_t, bool, bool, int, int, int, double *, int, double *,
                            double *, int, int, double *, int, int *);

} // namespace calaman::device
