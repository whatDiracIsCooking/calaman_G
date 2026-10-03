// laqr5.cu
//
// The device-kernel half of calaman.laqr5: a single-thread port of LAPACK's
// ?laqr5 in its non-accumulated (KACC22 = 0) form, launched through
// wwr.extension.parallel_for over a grid of one. ?laqr5 chases chains of
// double-implicit-shift bulges through an isolated diagonal block of an
// upper-Hessenberg matrix -- the inner engine of the multishift QR sweep
// (?laqr0 calls it). The per-reflector work is O(1), but each order-2/3
// reflector is applied to O(N) rows/columns of H (and Z), so one thread runs
// the whole sweep with no host round-trip, the shape of calaman.laexc.
//
// FAITHFUL TO THE REFERENCE, by design: the oracle is reference ?laqr5 called
// with KACC22 = 0, so the structure mirrors dlaqr5.f line for line for that
// path -- the shift shuffle into real/complex pairs, the NBMPS-bulge chain over
// INCOL/KRCOL, the bottom 2-by-2 bulge special case (BMP22), the normal 3-by-3
// chain with its delayed-row transform and bulge-collapse reintroduction, the
// vigilant-deflation test (traditional + Ahues-Tisseur), and the from-the-left
// column updates. The KACC22 >= 1 accumulation (U, the DGEMM far-from-diagonal
// multiplies and WV/WH workspace) is deliberately NOT ported: the non-
// accumulated sweep is complete on its own, and the test pins the oracle to
// KACC22 = 0 so device and reference take the same arithmetic path.
//
// The aux routines ?laqr1 (first column of the shift polynomial) and ?larfg
// (order-2/3 reflector generation) are inlined here as __device__ helpers so
// the whole sweep is one kernel; ?laqr1's arithmetic matches calaman.laqr1 and
// the reference verbatim. LAPACK's DLAMCH('S') (safe minimum) is FLT_MIN /
// DBL_MIN and DLAMCH('E'/'P') (eps) is FLT_EPSILON / DBL_EPSILON -- the <cfloat>
// macros, usable on the device where std::numeric_limits is not. Indices track
// the Fortran 1-based scheme; the column-major (i,j) accessors fold the leading
// dimension out.
//
// REAL ONLY: the surface is float / double, matching ?slaqr5 / ?dlaqr5 (the
// complex ?laqr5 differs). Shared unchanged between both backends: a .cu is
// compiled by the backend compiler, and parallel_for's launch machinery plus
// the precision-neutral math (sqrt/fabs/hypot/copysign) arrive through device
// headers.
#include "laqr5_bridge.h"

#include "extension/parallel_for/parallel_for.cuh"
#include "wrappers/math/math.cuh"

#include <cfloat>
#include <cstddef>

namespace calaman::device {

namespace {

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

// ------------------------------------------------------------------ DLAQR1
//
// First column of the shift polynomial (H - s1*I)(H - s2*I), reference ?laqr1
// verbatim (N is 2 or 3). Writes the length-N result into v[0..N-1]; the scale
// S keeps v[0] modest. Matches calaman.laqr1's arithmetic exactly.
template<typename T>
__device__ void dlaqr1(const int n, const T *h, const int ldh, const T sr1, const T si1,
                       const T sr2, const T si2, T *v) {
  const T zero{0};
  auto H = [=](int i, int j) -> T { return h[(i - 1) + (j - 1) * ldh]; };
  if (n == 2) {
    const T s = dabs(H(1, 1) - sr2) + dabs(si2) + dabs(H(2, 1));
    if (s == zero) {
      v[0] = zero;
      v[1] = zero;
    } else {
      const T h21s = H(2, 1) / s;
      v[0] = h21s * H(1, 2) + (H(1, 1) - sr1) * ((H(1, 1) - sr2) / s) - si1 * (si2 / s);
      v[1] = h21s * (H(1, 1) + H(2, 2) - sr1 - sr2);
    }
  } else {
    const T s = dabs(H(1, 1) - sr2) + dabs(si2) + dabs(H(2, 1)) + dabs(H(3, 1));
    if (s == zero) {
      v[0] = zero;
      v[1] = zero;
      v[2] = zero;
    } else {
      const T h21s = H(2, 1) / s;
      const T h31s = H(3, 1) / s;
      v[0] = (H(1, 1) - sr1) * ((H(1, 1) - sr2) / s) - si1 * (si2 / s) + H(1, 2) * h21s +
             H(1, 3) * h31s;
      v[1] = h21s * (H(1, 1) + H(2, 2) - sr1 - sr2) + H(2, 3) * h31s;
      v[2] = h31s * (H(1, 1) + H(3, 3) - sr1 - sr2) + h21s * H(3, 2);
    }
  }
}

// ------------------------------------------------------------------ DLARFG
//
// Generate the order-n Householder reflector H = I - tau*v*v^T with v = [1; x]
// (x the n-1 tail) such that H*[alpha; x] = [beta; 0], reference ?larfg verbatim,
// including the safmin rescaling loop. On entry alpha is x[0] and x the length
// n-1 tail (unit stride); on return alpha holds beta, x holds the reflector tail,
// and tau is returned. ?laqr5 calls this with n == 2 and n == 3.
template<typename T>
__device__ T dlarfg(const int n, T &alpha, T *x) {
  if (n <= 1) {
    return T{0};
  }
  const int m = n - 1;
  // DNRM2(m, x): sqrt(sum x_i^2), via hypot for the m<=2 cases ?laqr5 uses.
  auto nrm2 = [&]() -> T {
    if (m == 1) {
      return dabs(x[0]);
    }
    return wwr::hypot(x[0], x[1]); // m == 2
  };
  T xnorm = nrm2();
  if (xnorm == T{0}) {
    return T{0};
  }
  // DLAPY2(alpha, xnorm) = hypot; BETA = -sign(DLAPY2, alpha).
  T beta = -wwr::copysign(wwr::hypot(alpha, xnorm), alpha);
  const T safmin = lamch<T>::sfmin() / lamch<T>::eps();
  int knt = 0;
  if (dabs(beta) < safmin) {
    const T rsafmn = T{1} / safmin;
    do {
      ++knt;
      for (int i = 0; i < m; ++i) {
        x[i] *= rsafmn;
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
    x[i] *= scal;
  }
  for (int j = 0; j < knt; ++j) {
    beta *= safmin;
  }
  alpha = beta;
  return tau;
}

// The whole sweep, run by one thread. Members are const scalar / pointer, as
// device_functor requires: trivially copyable, and the const deletes the
// copy-assignment the grid-constant kernel copy would otherwise permit. The
// index is ignored (the grid is a single element).
template<typename T>
struct Laqr5Functor {
  const bool wantt_;
  const bool wantz_;
  const int n_;
  const int ktop_;
  const int kbot_;
  const int nshfts_;
  T *const sr_;
  T *const si_;
  T *const h_;
  const int ldh_;
  const int iloz_;
  const int ihiz_;
  T *const z_;
  const int ldz_;

  __device__ void operator()(std::size_t) const {
    const T zero{0}, one{1};
    const int n = n_;
    const int ktop = ktop_;
    const int kbot = kbot_;
    const int ldh = ldh_;
    const int ldz = ldz_;
    const bool wantt = wantt_;
    const bool wantz = wantz_;
    const int iloz = iloz_;
    const int ihiz = ihiz_;
    T *const sr = sr_;
    T *const si = si_;
    T *const h = h_;
    T *const z = z_;

    // 1-based column-major references into H and Z, matching the Fortran.
    auto H = [=](int i, int j) -> T & {
      return h[(i - 1) + static_cast<std::ptrdiff_t>(j - 1) * ldh];
    };
    auto Z = [=](int i, int j) -> T & {
      return z[(i - 1) + static_cast<std::ptrdiff_t>(j - 1) * ldz];
    };
    // 1-based shift access.
    auto SR = [=](int i) -> T & { return sr[i - 1]; };
    auto SI = [=](int i) -> T & { return si[i - 1]; };

    if (nshfts_ < 2) {
      return;
    }
    if (ktop >= kbot) {
      return;
    }

    // Shuffle shifts into real pairs and complex-conjugate pairs.
    for (int i = 1; i <= nshfts_ - 2; i += 2) {
      if (SI(i) != -SI(i + 1)) {
        T swap = SR(i);
        SR(i) = SR(i + 1);
        SR(i + 1) = SR(i + 2);
        SR(i + 2) = swap;
        swap = SI(i);
        SI(i) = SI(i + 1);
        SI(i + 1) = SI(i + 2);
        SI(i + 2) = swap;
      }
    }

    const int ns = nshfts_ - (nshfts_ % 2);

    const T safmin = lamch<T>::sfmin();
    const T ulp = lamch<T>::eps();
    const T smlnum = safmin * (static_cast<T>(n) / ulp);

    // KACC22 = 0: ACCUM is always false.
    if (ktop + 2 <= kbot) {
      H(ktop + 2, ktop) = zero;
    }

    const int nbmps = ns / 2;

    // Per-bulge reflector columns V(1:3, 1:nbmps), column-major, and VT(3).
    // nbmps <= nshfts/2; cap the on-stack scratch at a generous bound.
    constexpr int kMaxBulges = 64;
    T vcol[3 * kMaxBulges];
    auto V = [&](int r, int m) -> T & { return vcol[(r - 1) + (m - 1) * 3]; };
    T vt[3];

    for (int incol = ktop - 2 * nbmps + 1; incol <= kbot - 2; incol += 2 * nbmps) {
      // JTOP: index from which right updates start (ACCUM false).
      const int jtop = wantt ? 1 : ktop;
      const int ndcol = incol + 4 * nbmps; // KDU = 4*NBMPS

      const int krcol_end = dmin(incol + 2 * nbmps - 1, kbot - 2);
      for (int krcol = incol; krcol <= krcol_end; ++krcol) {
        const int mtop = dmax(1, (ktop - krcol) / 2 + 1);
        const int mbot = dmin(nbmps, (kbot - krcol - 1) / 2);
        const int m22 = mbot + 1;
        const bool bmp22 = (mbot < nbmps) && (krcol + 2 * (m22 - 1) == kbot - 2);

        // ==== Bottom 2-by-2 bulge special case. ====
        if (bmp22) {
          const int k = krcol + 2 * (m22 - 1);
          if (k == ktop - 1) {
            dlaqr1(2, &H(k + 1, k + 1), ldh, SR(2 * m22 - 1), SI(2 * m22 - 1), SR(2 * m22),
                   SI(2 * m22), &V(1, m22));
            T beta = V(1, m22);
            T xv[1] = {V(2, m22)};
            V(1, m22) = dlarfg(2, beta, xv);
            V(2, m22) = xv[0];
          } else {
            T beta = H(k + 1, k);
            T xv[1] = {H(k + 2, k)};
            V(1, m22) = dlarfg(2, beta, xv);
            V(2, m22) = xv[0];
            H(k + 1, k) = beta;
            H(k + 2, k) = zero;
          }

          // Update from the right within the window.
          {
            const T t1 = V(1, m22);
            const T t2 = t1 * V(2, m22);
            const int jhi = dmin(kbot, k + 3);
            for (int j = jtop; j <= jhi; ++j) {
              const T refsum = H(j, k + 1) + V(2, m22) * H(j, k + 2);
              H(j, k + 1) -= refsum * t1;
              H(j, k + 2) -= refsum * t2;
            }
          }
          // Update from the left within the window.
          {
            const int jbot = wantt ? n : kbot;
            const T t1 = V(1, m22);
            const T t2 = t1 * V(2, m22);
            for (int j = k + 1; j <= jbot; ++j) {
              const T refsum = H(k + 1, j) + V(2, m22) * H(k + 2, j);
              H(k + 1, j) -= refsum * t1;
              H(k + 2, j) -= refsum * t2;
            }
          }
          // Vigilant deflation.
          if (k >= ktop) {
            if (H(k + 1, k) != zero) {
              T tst1 = dabs(H(k, k)) + dabs(H(k + 1, k + 1));
              if (tst1 == zero) {
                if (k >= ktop + 1) {
                  tst1 += dabs(H(k, k - 1));
                }
                if (k >= ktop + 2) {
                  tst1 += dabs(H(k, k - 2));
                }
                if (k >= ktop + 3) {
                  tst1 += dabs(H(k, k - 3));
                }
                if (k <= kbot - 2) {
                  tst1 += dabs(H(k + 2, k + 1));
                }
                if (k <= kbot - 3) {
                  tst1 += dabs(H(k + 3, k + 1));
                }
                if (k <= kbot - 4) {
                  tst1 += dabs(H(k + 4, k + 1));
                }
              }
              if (dabs(H(k + 1, k)) <= dmax(smlnum, ulp * tst1)) {
                const T h12 = dmax(dabs(H(k + 1, k)), dabs(H(k, k + 1)));
                const T h21 = dmin(dabs(H(k + 1, k)), dabs(H(k, k + 1)));
                const T h11 = dmax(dabs(H(k + 1, k + 1)), dabs(H(k, k) - H(k + 1, k + 1)));
                const T h22 = dmin(dabs(H(k + 1, k + 1)), dabs(H(k, k) - H(k + 1, k + 1)));
                const T scl = h11 + h12;
                const T tst2 = h22 * (h11 / scl);
                if (tst2 == zero || h21 * (h12 / scl) <= dmax(smlnum, ulp * tst2)) {
                  H(k + 1, k) = zero;
                }
              }
            }
          }
          // Accumulate into Z (ACCUM false, so the WANTZ branch applies).
          if (wantz) {
            const T t1 = V(1, m22);
            const T t2 = t1 * V(2, m22);
            for (int j = iloz; j <= ihiz; ++j) {
              const T refsum = Z(j, k + 1) + V(2, m22) * Z(j, k + 2);
              Z(j, k + 1) -= refsum * t1;
              Z(j, k + 2) -= refsum * t2;
            }
          }
        }

        // ==== Normal case: chain of 3-by-3 reflections. ====
        for (int m = mbot; m >= mtop; --m) {
          const int k = krcol + 2 * (m - 1);
          if (k == ktop - 1) {
            dlaqr1(3, &H(ktop, ktop), ldh, SR(2 * m - 1), SI(2 * m - 1), SR(2 * m), SI(2 * m),
                   &V(1, m));
            T alpha = V(1, m);
            T xv[2] = {V(2, m), V(3, m)};
            V(1, m) = dlarfg(3, alpha, xv);
            V(2, m) = xv[0];
            V(3, m) = xv[1];
          } else {
            // Delayed transform of row below the Mth bulge (first two cols zero).
            const T t1d = V(1, m);
            const T t2d = t1d * V(2, m);
            const T t3d = t1d * V(3, m);
            const T refsum = V(3, m) * H(k + 3, k + 2);
            H(k + 3, k) = -refsum * t1d;
            H(k + 3, k + 1) = -refsum * t2d;
            H(k + 3, k + 2) -= refsum * t3d;

            // Reflection to move the Mth bulge one step.
            T beta = H(k + 1, k);
            T xv[2] = {H(k + 2, k), H(k + 3, k)};
            T tau = dlarfg(3, beta, xv);
            V(1, m) = tau;
            V(2, m) = xv[0];
            V(3, m) = xv[1];

            // Bulge may collapse: reinflate with the two-small-subdiagonals trick.
            if (H(k + 3, k) != zero || H(k + 3, k + 1) != zero || H(k + 3, k + 2) == zero) {
              H(k + 1, k) = beta;
              H(k + 2, k) = zero;
              H(k + 3, k) = zero;
            } else {
              dlaqr1(3, &H(k + 1, k + 1), ldh, SR(2 * m - 1), SI(2 * m - 1), SR(2 * m), SI(2 * m),
                     vt);
              T alpha = vt[0];
              T xvt[2] = {vt[1], vt[2]};
              T taut = dlarfg(3, alpha, xvt);
              vt[0] = taut;
              vt[1] = xvt[0];
              vt[2] = xvt[1];
              const T t1 = vt[0];
              const T t2 = t1 * vt[1];
              const T t3 = t1 * vt[2];
              const T rs = H(k + 1, k) + vt[1] * H(k + 2, k);
              if (dabs(H(k + 2, k) - rs * t2) + dabs(rs * t3) >
                  ulp * (dabs(H(k, k)) + dabs(H(k + 1, k + 1)) + dabs(H(k + 2, k + 2)))) {
                H(k + 1, k) = beta;
                H(k + 2, k) = zero;
                H(k + 3, k) = zero;
              } else {
                H(k + 1, k) -= rs * t1;
                H(k + 2, k) = zero;
                H(k + 3, k) = zero;
                V(1, m) = vt[0];
                V(2, m) = vt[1];
                V(3, m) = vt[2];
              }
            }
          }

          // Apply reflection from the right and first column of the left update.
          const T t1 = V(1, m);
          const T t2 = t1 * V(2, m);
          const T t3 = t1 * V(3, m);
          const int jhi = dmin(kbot, k + 3);
          for (int j = jtop; j <= jhi; ++j) {
            const T refsum = H(j, k + 1) + V(2, m) * H(j, k + 2) + V(3, m) * H(j, k + 3);
            H(j, k + 1) -= refsum * t1;
            H(j, k + 2) -= refsum * t2;
            H(j, k + 3) -= refsum * t3;
          }
          {
            const T refsum = H(k + 1, k + 1) + V(2, m) * H(k + 2, k + 1) + V(3, m) * H(k + 3, k + 1);
            H(k + 1, k + 1) -= refsum * t1;
            H(k + 2, k + 1) -= refsum * t2;
            H(k + 3, k + 1) -= refsum * t3;
          }

          // Vigilant deflation (skip when K < KTOP).
          if (k < ktop) {
            continue;
          }
          if (H(k + 1, k) != zero) {
            T tst1 = dabs(H(k, k)) + dabs(H(k + 1, k + 1));
            if (tst1 == zero) {
              if (k >= ktop + 1) {
                tst1 += dabs(H(k, k - 1));
              }
              if (k >= ktop + 2) {
                tst1 += dabs(H(k, k - 2));
              }
              if (k >= ktop + 3) {
                tst1 += dabs(H(k, k - 3));
              }
              if (k <= kbot - 2) {
                tst1 += dabs(H(k + 2, k + 1));
              }
              if (k <= kbot - 3) {
                tst1 += dabs(H(k + 3, k + 1));
              }
              if (k <= kbot - 4) {
                tst1 += dabs(H(k + 4, k + 1));
              }
            }
            if (dabs(H(k + 1, k)) <= dmax(smlnum, ulp * tst1)) {
              const T h12 = dmax(dabs(H(k + 1, k)), dabs(H(k, k + 1)));
              const T h21 = dmin(dabs(H(k + 1, k)), dabs(H(k, k + 1)));
              const T h11 = dmax(dabs(H(k + 1, k + 1)), dabs(H(k, k) - H(k + 1, k + 1)));
              const T h22 = dmin(dabs(H(k + 1, k + 1)), dabs(H(k, k) - H(k + 1, k + 1)));
              const T scl = h11 + h12;
              const T tst2 = h22 * (h11 / scl);
              if (tst2 == zero || h21 * (h12 / scl) <= dmax(smlnum, ulp * tst2)) {
                H(k + 1, k) = zero;
              }
            }
          }
        } // end 3-by-3 chain

        // Multiply H by reflections from the left (delayed columns).
        const int jbot = wantt ? n : kbot;
        for (int m = mbot; m >= mtop; --m) {
          const int k = krcol + 2 * (m - 1);
          const T t1 = V(1, m);
          const T t2 = t1 * V(2, m);
          const T t3 = t1 * V(3, m);
          const int jlo = dmax(ktop, krcol + 2 * m);
          for (int j = jlo; j <= jbot; ++j) {
            const T refsum = H(k + 1, j) + V(2, m) * H(k + 2, j) + V(3, m) * H(k + 3, j);
            H(k + 1, j) -= refsum * t1;
            H(k + 2, j) -= refsum * t2;
            H(k + 3, j) -= refsum * t3;
          }
        }

        // Accumulate into Z from the right (ACCUM false: WANTZ branch).
        if (wantz) {
          for (int m = mbot; m >= mtop; --m) {
            const int k = krcol + 2 * (m - 1);
            const T t1 = V(1, m);
            const T t2 = t1 * V(2, m);
            const T t3 = t1 * V(3, m);
            for (int j = iloz; j <= ihiz; ++j) {
              const T refsum = Z(j, k + 1) + V(2, m) * Z(j, k + 2) + V(3, m) * Z(j, k + 3);
              Z(j, k + 1) -= refsum * t1;
              Z(j, k + 2) -= refsum * t2;
              Z(j, k + 3) -= refsum * t3;
            }
          }
        }
      } // end krcol
      (void)ndcol; // NDCOL only feeds the ACCUM path, not ported here.
    } // end incol
  }
};

} // namespace

template<typename T>
void laqr5(const wwr::wwrStream_t stream, const bool wantt, const bool wantz, const int n,
           const int ktop, const int kbot, const int nshfts, T *const sr, T *const si, T *const h,
           const int ldh, const int iloz, const int ihiz, T *const z, const int ldz) {
  const Laqr5Functor<T> functor{wantt, wantz, n,    ktop, kbot, nshfts, sr,
                                 si,    h,     ldh,  iloz, ihiz, z,      ldz};
  wwr::extension::parallel_for<std::size_t>(stream, std::size_t{1}, functor);
}

// One per supported precision, matching laqr5_bridge.h's declaration and
// interface.cppm's extern-template list -- a type added here without being added
// there, or vice versa, links against nothing.
template void laqr5<float>(wwr::wwrStream_t, bool, bool, int, int, int, int, float *, float *,
                           float *, int, int, int, float *, int);
template void laqr5<double>(wwr::wwrStream_t, bool, bool, int, int, int, int, double *, double *,
                            double *, int, int, int, double *, int);

} // namespace calaman::device
