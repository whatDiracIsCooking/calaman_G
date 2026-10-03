// trevc3.cu
//
// The device-kernel half of calaman.trevc3: a single-thread port of LAPACK's
// ?trevc3 for HOWMNY = 'A' with the eigenvectors written straight into VL / VR
// (no OVER back-transform), launched through wwr.extension.parallel_for over a
// grid of one. ?trevc3 computes the left and/or right eigenvectors of an upper
// quasi-triangular (real Schur) matrix T by back-substitution: each eigenvalue
// -- real (a 1-by-1 diagonal block) or a complex-conjugate pair (a 2-by-2
// block) -- yields one column of VR (right) and/or VL (left). The per-block
// solve is O(1) and each back-substitution AXPY/DOT touches O(N) entries of the
// growing vector, so one thread runs the whole computation with no host
// round-trip, the shape of calaman.laexc and calaman.laqr5.
//
// FAITHFUL TO THE REFERENCE, by design: the oracle is reference ?trevc3, so the
// structure mirrors dtrevc3.f for HOWMNY = 'A' -- the real/complex split off the
// subdiagonal, the right-vector downward sweep (columns KI = N..1) seeding
// WORK(KI) = 1 and WORK(1:KI-1) = -T(1:KI-1,KI), the left-vector upward sweep
// (columns KI = 1..N) over the transposed trailing block, the per-block ?laln2
// solve with its SCALE / XNORM overflow guards, and the final infinity-norm
// normalization (IDAMAX then DSCAL by 1/|v_max|). The blocked back-transform
// (HOWMNY = 'B' / OVER, the DGEMM column-block multiplies that make ?trevc3 the
// level-3 ?trevc) is NOT ported: eigenvectors of T are complete on their own,
// the test pins the oracle to HOWMNY = 'A', and both paths normalize the SAME
// way, so device and reference agree up to the sign the normalization fixes.
//
// ?laln2 (the scaled 1-by-1 / 2-by-2 solve each block needs) is inlined here as
// a __device__ helper -- calaman.laln2 is a host module whose kernel is a
// launch functor, unusable from inside this kernel -- so, like laexc inlining
// its auxiliaries, this links no sibling LAPACK module. The inlined solve's
// complex divisions reach the SAME header-only host/device ladiv_scalar from
// "ladiv/ladiv.h" the reference reaches as DLADIV, included root-relative so the
// call has device linkage. LAPACK's DLAMCH('S') is FLT_MIN / DBL_MIN and
// DLAMCH('P') is FLT_EPSILON / DBL_EPSILON -- the <cfloat> macros, usable on the
// device where std::numeric_limits is not. Indices track the Fortran 1-based
// scheme; the column-major (i,j) accessors fold the leading dimension out.
//
// REAL ONLY: the surface is float / double, matching ?strevc3 / ?dtrevc3 (the
// complex ?trevc3 differs). Shared unchanged between both backends: a .cu is
// compiled by the backend compiler, and parallel_for's launch machinery arrives
// through its device header.
#include "trevc3_bridge.h"

#include "ladiv/ladiv.h"

#include "extension/parallel_for/parallel_for.cuh"
#include "wrappers/math/math.cuh"

#include <cfloat>
#include <cstddef>

namespace calaman::device {

namespace {

// abs / max / min in exact arithmetic (no rounding) so they match LAPACK's
// intrinsic ABS / MAX / MIN bit for bit -- the control flow (pivot choice,
// SMINI tests) must.
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

// LAPACK's DLAMCH constants, device-side: sfmin() is DLAMCH('S') (safe minimum),
// eps() is DLAMCH('P'/'E') (relative machine precision). std::numeric_limits is
// a __host__ constexpr the device pass rejects, so these read <cfloat>.
template<typename T>
struct lamch;
template<>
struct lamch<float> {
  static __device__ __forceinline__ float sfmin() { return FLT_MIN; }
  static __device__ __forceinline__ float eps() { return FLT_EPSILON; }
};
template<>
struct lamch<double> {
  static __device__ __forceinline__ double sfmin() { return DBL_MIN; }
  static __device__ __forceinline__ double eps() { return DBL_EPSILON; }
};

// Inlined ?laln2: solve (ca*op(A) - w*D) X = SCALE*B for the order-NA (1 or 2)
// real system, w = wr + i*wi real (nw == 1) or complex (nw == 2). A port of
// dlaln2.f's arithmetic -- the same complete-pivoting order, SMINI perturbation
// and scale/xnorm formulas -- matching calaman.laln2's kernel line for line
// (that module's oracle pins it to the reference). a/b/x are small column-major
// slabs in registers/local memory with the given leading dimensions; scale and
// xnorm are returned by pointer. See src/laln2/laln2.cu for the shared source.
template<typename T>
__device__ void laln2_solve(const bool ltrans, const int na, const int nw, const T smin, const T ca,
                            const T *const a, const int lda, const T d1, const T d2,
                            const T *const b, const int ldb, const T wr, const T wi, T *const x,
                            const int ldx, T *const scale_out, T *const xnorm_out) {
  const T zero = T{0};
  const T one = T{1};
  const T two = T{2};

  const T smlnum = two * lamch<T>::sfmin();
  const T bignum = one / smlnum;
  const T smini = dmax(smin, smlnum);

  auto A = [=](int i, int j) -> T { return a[(i - 1) + (j - 1) * lda]; };
  auto B = [=](int i, int j) -> T { return b[(i - 1) + (j - 1) * ldb]; };
  auto setX = [=](int i, int j, T v) { x[(i - 1) + (j - 1) * ldx] = v; };

  T scale = one;

  const bool rswap[5] = {false, false, true, false, true};
  const bool zswap[5] = {false, false, false, true, true};
  const int ipivot[5][5] = {
      {0, 0, 0, 0, 0}, {0, 1, 2, 3, 4}, {0, 2, 1, 4, 3}, {0, 3, 4, 1, 2}, {0, 4, 3, 2, 1}};

  if (na == 1) {
    if (nw == 1) {
      T csr = ca * A(1, 1) - wr * d1;
      T cnorm = dabs(csr);
      if (cnorm < smini) {
        csr = smini;
        cnorm = smini;
      }
      const T bnorm = dabs(B(1, 1));
      if (cnorm < one && bnorm > one) {
        if (bnorm > bignum * cnorm) {
          scale = one / bnorm;
        }
      }
      const T xr = (B(1, 1) * scale) / csr;
      setX(1, 1, xr);
      *xnorm_out = dabs(xr);
    } else {
      T csr = ca * A(1, 1) - wr * d1;
      T csi = -wi * d1;
      T cnorm = dabs(csr) + dabs(csi);
      if (cnorm < smini) {
        csr = smini;
        csi = zero;
        cnorm = smini;
      }
      const T bnorm = dabs(B(1, 1)) + dabs(B(1, 2));
      if (cnorm < one && bnorm > one) {
        if (bnorm > bignum * cnorm) {
          scale = one / bnorm;
        }
      }
      T xr, xi;
      ladiv_scalar<T>(scale * B(1, 1), scale * B(1, 2), csr, csi, &xr, &xi);
      setX(1, 1, xr);
      setX(1, 2, xi);
      *xnorm_out = dabs(xr) + dabs(xi);
    }
    *scale_out = scale;
    return;
  }

  T crv[5];
  crv[1] = ca * A(1, 1) - wr * d1;
  crv[4] = ca * A(2, 2) - wr * d2;
  if (ltrans) {
    crv[3] = ca * A(2, 1);
    crv[2] = ca * A(1, 2);
  } else {
    crv[2] = ca * A(2, 1);
    crv[3] = ca * A(1, 2);
  }

  if (nw == 1) {
    T cmax = zero;
    int icmax = 0;
    for (int j = 1; j <= 4; ++j) {
      if (dabs(crv[j]) > cmax) {
        cmax = dabs(crv[j]);
        icmax = j;
      }
    }
    if (cmax < smini) {
      const T bnorm = dmax(dabs(B(1, 1)), dabs(B(2, 1)));
      if (smini < one && bnorm > one) {
        if (bnorm > bignum * smini) {
          scale = one / bnorm;
        }
      }
      const T temp = scale / smini;
      setX(1, 1, temp * B(1, 1));
      setX(2, 1, temp * B(2, 1));
      *xnorm_out = temp * bnorm;
      *scale_out = scale;
      return;
    }
    const T ur11 = crv[icmax];
    const T cr21 = crv[ipivot[2][icmax]];
    const T ur12 = crv[ipivot[3][icmax]];
    const T cr22 = crv[ipivot[4][icmax]];
    const T ur11r = one / ur11;
    const T lr21 = ur11r * cr21;
    T ur22 = cr22 - ur12 * lr21;
    if (dabs(ur22) < smini) {
      ur22 = smini;
    }
    T br1, br2;
    if (rswap[icmax]) {
      br1 = B(2, 1);
      br2 = B(1, 1);
    } else {
      br1 = B(1, 1);
      br2 = B(2, 1);
    }
    br2 = br2 - lr21 * br1;
    const T bbnd = dmax(dabs(br1 * (ur22 * ur11r)), dabs(br2));
    if (bbnd > one && dabs(ur22) < one) {
      if (bbnd >= bignum * dabs(ur22)) {
        scale = one / bbnd;
      }
    }
    const T xr2 = (br2 * scale) / ur22;
    const T xr1 = (scale * br1) * ur11r - xr2 * (ur11r * ur12);
    if (zswap[icmax]) {
      setX(1, 1, xr2);
      setX(2, 1, xr1);
    } else {
      setX(1, 1, xr1);
      setX(2, 1, xr2);
    }
    T xnorm = dmax(dabs(xr1), dabs(xr2));
    if (xnorm > one && cmax > one) {
      if (xnorm > bignum / cmax) {
        const T temp = cmax / bignum;
        setX(1, 1, temp * (zswap[icmax] ? xr2 : xr1));
        setX(2, 1, temp * (zswap[icmax] ? xr1 : xr2));
        xnorm = temp * xnorm;
        scale = temp * scale;
      }
    }
    *xnorm_out = xnorm;
    *scale_out = scale;
    return;
  }

  T civ[5];
  civ[1] = -wi * d1;
  civ[2] = zero;
  civ[3] = zero;
  civ[4] = -wi * d2;

  T cmax = zero;
  int icmax = 0;
  for (int j = 1; j <= 4; ++j) {
    if (dabs(crv[j]) + dabs(civ[j]) > cmax) {
      cmax = dabs(crv[j]) + dabs(civ[j]);
      icmax = j;
    }
  }
  if (cmax < smini) {
    const T bnorm = dmax(dabs(B(1, 1)) + dabs(B(1, 2)), dabs(B(2, 1)) + dabs(B(2, 2)));
    if (smini < one && bnorm > one) {
      if (bnorm > bignum * smini) {
        scale = one / bnorm;
      }
    }
    const T temp = scale / smini;
    setX(1, 1, temp * B(1, 1));
    setX(2, 1, temp * B(2, 1));
    setX(1, 2, temp * B(1, 2));
    setX(2, 2, temp * B(2, 2));
    *xnorm_out = temp * bnorm;
    *scale_out = scale;
    return;
  }
  const T ur11 = crv[icmax];
  const T ui11 = civ[icmax];
  const T cr21 = crv[ipivot[2][icmax]];
  const T ci21 = civ[ipivot[2][icmax]];
  const T ur12 = crv[ipivot[3][icmax]];
  const T ui12 = civ[ipivot[3][icmax]];
  const T cr22 = crv[ipivot[4][icmax]];
  const T ci22 = civ[ipivot[4][icmax]];
  T ur11r, ui11r, lr21, li21, ur12s, ui12s, ur22, ui22;
  if (icmax == 1 || icmax == 4) {
    if (dabs(ur11) > dabs(ui11)) {
      const T temp = ui11 / ur11;
      ur11r = one / (ur11 * (one + temp * temp));
      ui11r = -temp * ur11r;
    } else {
      const T temp = ur11 / ui11;
      ui11r = -one / (ui11 * (one + temp * temp));
      ur11r = -temp * ui11r;
    }
    lr21 = cr21 * ur11r;
    li21 = cr21 * ui11r;
    ur12s = ur12 * ur11r;
    ui12s = ur12 * ui11r;
    ur22 = cr22 - ur12 * lr21;
    ui22 = ci22 - ur12 * li21;
  } else {
    ur11r = one / ur11;
    ui11r = zero;
    lr21 = cr21 * ur11r;
    li21 = ci21 * ur11r;
    ur12s = ur12 * ur11r;
    ui12s = ui12 * ur11r;
    ur22 = cr22 - ur12 * lr21 + ui12 * li21;
    ui22 = -ur12 * li21 - ui12 * lr21;
  }
  T u22abs = dabs(ur22) + dabs(ui22);
  if (u22abs < smini) {
    ur22 = smini;
    ui22 = zero;
  }
  T br1, br2, bi1, bi2;
  if (rswap[icmax]) {
    br2 = B(1, 1);
    br1 = B(2, 1);
    bi2 = B(1, 2);
    bi1 = B(2, 2);
  } else {
    br1 = B(1, 1);
    br2 = B(2, 1);
    bi1 = B(1, 2);
    bi2 = B(2, 2);
  }
  br2 = br2 - lr21 * br1 + li21 * bi1;
  bi2 = bi2 - li21 * br1 - lr21 * bi1;
  const T bbnd = dmax((dabs(br1) + dabs(bi1)) * (u22abs * (dabs(ur11r) + dabs(ui11r))),
                      dabs(br2) + dabs(bi2));
  if (bbnd > one && u22abs < one) {
    if (bbnd >= bignum * u22abs) {
      scale = one / bbnd;
      br1 = scale * br1;
      bi1 = scale * bi1;
      br2 = scale * br2;
      bi2 = scale * bi2;
    }
  }
  T xr2, xi2;
  ladiv_scalar<T>(br2, bi2, ur22, ui22, &xr2, &xi2);
  const T xr1 = ur11r * br1 - ui11r * bi1 - ur12s * xr2 + ui12s * xi2;
  const T xi1 = ui11r * br1 + ur11r * bi1 - ui12s * xr2 - ur12s * xi2;
  if (zswap[icmax]) {
    setX(1, 1, xr2);
    setX(2, 1, xr1);
    setX(1, 2, xi2);
    setX(2, 2, xi1);
  } else {
    setX(1, 1, xr1);
    setX(2, 1, xr2);
    setX(1, 2, xi1);
    setX(2, 2, xi2);
  }
  T xnorm = dmax(dabs(xr1) + dabs(xi1), dabs(xr2) + dabs(xi2));
  if (xnorm > one && cmax > one) {
    if (xnorm > bignum / cmax) {
      const T temp = cmax / bignum;
      setX(1, 1, temp * (zswap[icmax] ? xr2 : xr1));
      setX(2, 1, temp * (zswap[icmax] ? xr1 : xr2));
      setX(1, 2, temp * (zswap[icmax] ? xi2 : xi1));
      setX(2, 2, temp * (zswap[icmax] ? xi1 : xi2));
      xnorm = temp * xnorm;
      scale = temp * scale;
    }
  }
  *xnorm_out = xnorm;
  *scale_out = scale;
}

// The per-call eigenvector computation. Members are const scalar / pointer, as
// device_functor requires: trivially copyable, the const deleting the
// copy-assignment the grid-constant kernel copy would otherwise permit. One
// thread runs operator(); the index is ignored (the grid is a single element).
template<typename T>
struct Trevc3Functor {
  const bool want_left_;
  const bool want_right_;
  const int n_;
  const T *const t_;
  const int ldt_;
  T *const vl_;
  const int ldvl_;
  T *const vr_;
  const int ldvr_;
  T *const work_;

  __device__ void operator()(std::size_t) const {
    const T zero = T{0};
    const T one = T{1};
    const int n = n_;
    const int ldt = ldt_;
    const T *const t = t_;
    T *const work = work_;

    // Column-major 1-based accessor on T (read-only). WORK holds up to three
    // length-N columns; dtrevc3's non-blocked path uses WORK(:,2) (right) and
    // WORK(:,3) (left) -- here w1/w2 index two of them, 1-based within a column.
    auto Tm = [=](int i, int j) -> T { return t[(i - 1) + (j - 1) * ldt]; };
    T *const w1 = work;         // first candidate column (real part)
    T *const w2 = work + n;     // second candidate column (imaginary part)

    const T ulp = lamch<T>::eps();
    const T smlnum = lamch<T>::sfmin() * (static_cast<T>(n) / ulp);
    const T bignum = (one - ulp) / lamch<T>::sfmin();

    if (want_right_) {
      int is = n; // output column (eigenvectors fill VR right to left)
      // ip encodes the block: 0 real, -1 at the 2nd row of a complex pair.
      int ip = 0;
      for (int ki = n; ki >= 1; --ki) {
        if (ip == -1) {
          ip = 0;
          continue; // the complex pair was handled at ki+1
        }
        if (ki == 1) {
          ip = 0;
        } else if (Tm(ki, ki - 1) != zero) {
          ip = -1;
        } else {
          ip = 0;
        }

        // Eigenvalue(s) of the diagonal block.
        T wr, wi;
        if (ip == 0) {
          wr = Tm(ki, ki);
          wi = zero;
        } else {
          // Standardized 2-by-2 block at (ki-1,ki-1): equal diagonals, so
          // w = T(ki,ki) +/- i*sqrt(|T(ki,ki-1)|*|T(ki-1,ki)|), as dtrevc3.
          wr = Tm(ki, ki);
          wi = std::sqrt(dabs(Tm(ki, ki - 1))) * std::sqrt(dabs(Tm(ki - 1, ki)));
        }
        const T smin = dmax(ulp * (dabs(wr) + dabs(wi)), smlnum);

        if (ip == 0) {
          // --- Real right eigenvector ---
          w1[ki - 1] = one;
          for (int k = 1; k <= ki - 1; ++k) {
            w1[k - 1] = -Tm(k, ki);
          }
          // Back-substitution: (T(1:ki-1,1:ki-1) - wr) upper quasi-triangular.
          int jnxt = ki - 1;
          for (int j = ki - 1; j >= 1; --j) {
            if (j > jnxt) {
              continue;
            }
            int j1 = j, j2 = j;
            jnxt = j - 1;
            if (j > 1 && Tm(j, j - 1) != zero) {
              j1 = j - 1;
              jnxt = j - 2;
            }
            if (j1 == j2) {
              T x[2 * 2];
              T scale, xnorm;
              const T ablk = Tm(j, j);
              laln2_solve<T>(false, 1, 1, smin, one, &ablk, 1, one, one, &w1[j - 1], n, wr, zero, x,
                             2, &scale, &xnorm);
              // Rescale X to keep the RHS update in range (dtrevc3's guard).
              if (xnorm > one && w1[j - 1] > bignum / xnorm) {
                x[0] /= xnorm;
                scale /= xnorm;
              }
              if (scale != one) {
                for (int k = 1; k <= ki; ++k) {
                  w1[k - 1] *= scale;
                }
              }
              w1[j - 1] = x[0];
              // AXPY: WORK(1:j-1) -= x * T(1:j-1,j)
              for (int k = 1; k <= j - 1; ++k) {
                w1[k - 1] -= x[0] * Tm(k, j);
              }
            } else {
              T ablk[2 * 2];
              ablk[0] = Tm(j - 1, j - 1);
              ablk[1] = Tm(j, j - 1);
              ablk[2] = Tm(j - 1, j);
              ablk[3] = Tm(j, j);
              T x[2 * 2];
              T scale, xnorm;
              laln2_solve<T>(false, 2, 1, smin, one, ablk, 2, one, one, &w1[j - 2], n, wr, zero, x,
                             2, &scale, &xnorm);
              const T beta = dmax(w1[j - 2], w1[j - 1]);
              if (xnorm > one && beta > bignum / xnorm) {
                x[0] /= xnorm;
                x[1] /= xnorm;
                scale /= xnorm;
              }
              if (scale != one) {
                for (int k = 1; k <= ki; ++k) {
                  w1[k - 1] *= scale;
                }
              }
              w1[j - 2] = x[0];
              w1[j - 1] = x[1];
              for (int k = 1; k <= j - 2; ++k) {
                w1[k - 1] -= x[0] * Tm(k, j - 1) + x[1] * Tm(k, j);
              }
            }
          }
          // Copy to VR(:,is), normalize to unit infinity norm, zero the tail.
          int imax = 1;
          for (int k = 2; k <= ki; ++k) {
            if (dabs(w1[k - 1]) > dabs(w1[imax - 1])) {
              imax = k;
            }
          }
          const T remax = one / dabs(w1[imax - 1]);
          for (int k = 1; k <= ki; ++k) {
            vr_[(k - 1) + (is - 1) * ldvr_] = remax * w1[k - 1];
          }
          for (int k = ki + 1; k <= n; ++k) {
            vr_[(k - 1) + (is - 1) * ldvr_] = zero;
          }
          is -= 1;
        } else {
          // --- Complex right eigenvector pair (columns is-1 real, is imag) ---
          // Seed the two trailing components from the 2-by-2 block.
          if (dabs(Tm(ki - 1, ki)) >= dabs(Tm(ki, ki - 1))) {
            w1[ki - 2] = one;
            w2[ki - 1] = wi / Tm(ki - 1, ki);
          } else {
            w1[ki - 2] = -wi / Tm(ki, ki - 1);
            w2[ki - 1] = one;
          }
          w2[ki - 2] = zero;
          w1[ki - 1] = zero;
          for (int k = 1; k <= ki - 2; ++k) {
            w1[k - 1] = -w1[ki - 2] * Tm(k, ki - 1);
            w2[k - 1] = -w2[ki - 1] * Tm(k, ki);
          }
          int jnxt = ki - 2;
          for (int j = ki - 2; j >= 1; --j) {
            if (j > jnxt) {
              continue;
            }
            int j1 = j, j2 = j;
            jnxt = j - 1;
            if (j > 1 && Tm(j, j - 1) != zero) {
              j1 = j - 1;
              jnxt = j - 2;
            }
            if (j1 == j2) {
              T b[1 * 2];
              b[0] = w1[j - 1];
              b[1] = w2[j - 1];
              const T ablk = Tm(j, j);
              T x[2 * 2];
              T scale, xnorm;
              laln2_solve<T>(false, 1, 2, smin, one, &ablk, 1, one, one, b, 1, wr, wi, x, 2, &scale,
                             &xnorm);
              if (xnorm > one && w1[j - 1] > bignum / xnorm) {
                x[0] /= xnorm;
                x[2] /= xnorm;
                scale /= xnorm;
              }
              if (scale != one) {
                for (int k = 1; k <= ki; ++k) {
                  w1[k - 1] *= scale;
                  w2[k - 1] *= scale;
                }
              }
              w1[j - 1] = x[0];
              w2[j - 1] = x[2];
              for (int k = 1; k <= j - 1; ++k) {
                w1[k - 1] -= x[0] * Tm(k, j);
                w2[k - 1] -= x[2] * Tm(k, j);
              }
            } else {
              T ablk[2 * 2];
              ablk[0] = Tm(j - 1, j - 1);
              ablk[1] = Tm(j, j - 1);
              ablk[2] = Tm(j - 1, j);
              ablk[3] = Tm(j, j);
              T b[2 * 2];
              b[0] = w1[j - 2];
              b[1] = w1[j - 1];
              b[2] = w2[j - 2];
              b[3] = w2[j - 1];
              T x[2 * 2];
              T scale, xnorm;
              laln2_solve<T>(false, 2, 2, smin, one, ablk, 2, one, one, b, 2, wr, wi, x, 2, &scale,
                             &xnorm);
              const T beta = dmax(w1[j - 2], w1[j - 1]);
              if (xnorm > one && beta > bignum / xnorm) {
                const T rec = one / xnorm;
                x[0] *= rec;
                x[1] *= rec;
                x[2] *= rec;
                x[3] *= rec;
                scale *= rec;
              }
              if (scale != one) {
                for (int k = 1; k <= ki; ++k) {
                  w1[k - 1] *= scale;
                  w2[k - 1] *= scale;
                }
              }
              w1[j - 2] = x[0];
              w1[j - 1] = x[1];
              w2[j - 2] = x[2];
              w2[j - 1] = x[3];
              for (int k = 1; k <= j - 2; ++k) {
                w1[k - 1] -= x[0] * Tm(k, j - 1) + x[1] * Tm(k, j);
                w2[k - 1] -= x[2] * Tm(k, j - 1) + x[3] * Tm(k, j);
              }
            }
          }
          // Normalize the complex vector by max_k (|re_k| + |im_k|).
          T emax = zero;
          for (int k = 1; k <= ki; ++k) {
            emax = dmax(emax, dabs(w1[k - 1]) + dabs(w2[k - 1]));
          }
          const T remax = one / emax;
          for (int k = 1; k <= ki; ++k) {
            vr_[(k - 1) + (is - 2) * ldvr_] = remax * w1[k - 1];
            vr_[(k - 1) + (is - 1) * ldvr_] = remax * w2[k - 1];
          }
          for (int k = ki + 1; k <= n; ++k) {
            vr_[(k - 1) + (is - 2) * ldvr_] = zero;
            vr_[(k - 1) + (is - 1) * ldvr_] = zero;
          }
          is -= 2;
        }
      }
    }

    if (want_left_) {
      int is = 1; // output column (eigenvectors fill VL left to right)
      int ip = 0;
      for (int ki = 1; ki <= n; ++ki) {
        if (ip == 1) {
          ip = 0;
          continue;
        }
        if (ki == n) {
          ip = 0;
        } else if (Tm(ki + 1, ki) != zero) {
          ip = 1;
        } else {
          ip = 0;
        }

        T wr, wi;
        if (ip == 0) {
          wr = Tm(ki, ki);
          wi = zero;
        } else {
          wr = (Tm(ki, ki) + Tm(ki + 1, ki + 1)) * T{0.5};
          wi = std::sqrt(dabs(Tm(ki, ki + 1))) * std::sqrt(dabs(Tm(ki + 1, ki)));
        }
        const T smin = dmax(ulp * (dabs(wr) + dabs(wi)), smlnum);

        if (ip == 0) {
          // --- Real left eigenvector: solve (T(ki+1:n,ki+1:n)-wr)^T on top ---
          w1[ki - 1] = one;
          for (int k = ki + 1; k <= n; ++k) {
            w1[k - 1] = -Tm(ki, k);
          }
          T vmax = one;
          T vcrit = bignum;
          int jnxt = ki + 1;
          for (int j = ki + 1; j <= n; ++j) {
            if (j < jnxt) {
              continue;
            }
            int j1 = j, j2 = j;
            jnxt = j + 1;
            if (j < n && Tm(j + 1, j) != zero) {
              j2 = j + 1;
              jnxt = j + 2;
            }
            if (j1 == j2) {
              if (w1[j - 1] > vcrit) {
                const T rec = one / vmax;
                for (int k = ki; k <= n; ++k) {
                  w1[k - 1] *= rec;
                }
                vmax = one;
                vcrit = bignum;
              }
              // WORK(j) -= DOT(T(ki+1:j-1,j), WORK(ki+1:j-1))
              T dot = zero;
              for (int k = ki + 1; k <= j - 1; ++k) {
                dot += Tm(k, j) * w1[k - 1];
              }
              w1[j - 1] -= dot;
              const T ablk = Tm(j, j);
              T x[2 * 2];
              T scale, xnorm;
              laln2_solve<T>(false, 1, 1, smin, one, &ablk, 1, one, one, &w1[j - 1], n, wr, zero, x,
                             2, &scale, &xnorm);
              if (scale != one) {
                for (int k = ki; k <= n; ++k) {
                  w1[k - 1] *= scale;
                }
              }
              w1[j - 1] = x[0];
              vmax = dmax(dabs(w1[j - 1]), vmax);
              vcrit = bignum / vmax;
            } else {
              const T beta = dmax(w1[j - 1], w1[j]);
              if (beta > vcrit) {
                const T rec = one / vmax;
                for (int k = ki; k <= n; ++k) {
                  w1[k - 1] *= rec;
                }
                vmax = one;
                vcrit = bignum;
              }
              T dot1 = zero, dot2 = zero;
              for (int k = ki + 1; k <= j - 1; ++k) {
                dot1 += Tm(k, j) * w1[k - 1];
                dot2 += Tm(k, j + 1) * w1[k - 1];
              }
              w1[j - 1] -= dot1;
              w1[j] -= dot2;
              T ablk[2 * 2];
              ablk[0] = Tm(j, j);
              ablk[1] = Tm(j + 1, j);
              ablk[2] = Tm(j, j + 1);
              ablk[3] = Tm(j + 1, j + 1);
              T b[2];
              b[0] = w1[j - 1];
              b[1] = w1[j];
              T x[2 * 2];
              T scale, xnorm;
              laln2_solve<T>(true, 2, 1, smin, one, ablk, 2, one, one, b, 2, wr, zero, x, 2, &scale,
                             &xnorm);
              if (scale != one) {
                for (int k = ki; k <= n; ++k) {
                  w1[k - 1] *= scale;
                }
              }
              w1[j - 1] = x[0];
              w1[j] = x[1];
              vmax = dmax(dmax(dabs(w1[j - 1]), dabs(w1[j])), vmax);
              vcrit = bignum / vmax;
            }
          }
          int imax = ki;
          for (int k = ki + 1; k <= n; ++k) {
            if (dabs(w1[k - 1]) > dabs(w1[imax - 1])) {
              imax = k;
            }
          }
          const T remax = one / dabs(w1[imax - 1]);
          for (int k = ki; k <= n; ++k) {
            vl_[(k - 1) + (is - 1) * ldvl_] = remax * w1[k - 1];
          }
          for (int k = 1; k <= ki - 1; ++k) {
            vl_[(k - 1) + (is - 1) * ldvl_] = zero;
          }
          is += 1;
        } else {
          // --- Complex left eigenvector pair (columns is real, is+1 imag) ---
          if (dabs(Tm(ki, ki + 1)) >= dabs(Tm(ki + 1, ki))) {
            w1[ki - 1] = wi / Tm(ki, ki + 1);
            w2[ki] = one;
          } else {
            w1[ki - 1] = one;
            w2[ki] = -wi / Tm(ki + 1, ki);
          }
          w2[ki - 1] = zero;
          w1[ki] = zero;
          for (int k = ki + 2; k <= n; ++k) {
            w1[k - 1] = -w1[ki - 1] * Tm(ki, k);
            w2[k - 1] = -w2[ki] * Tm(ki + 1, k);
          }
          T vmax = one;
          T vcrit = bignum;
          int jnxt = ki + 2;
          for (int j = ki + 2; j <= n; ++j) {
            if (j < jnxt) {
              continue;
            }
            int j1 = j, j2 = j;
            jnxt = j + 1;
            if (j < n && Tm(j + 1, j) != zero) {
              j2 = j + 1;
              jnxt = j + 2;
            }
            if (j1 == j2) {
              if (w1[j - 1] > vcrit || w2[j - 1] > vcrit) {
                const T rec = one / vmax;
                for (int k = ki; k <= n; ++k) {
                  w1[k - 1] *= rec;
                  w2[k - 1] *= rec;
                }
                vmax = one;
                vcrit = bignum;
              }
              T d1 = zero, d2 = zero;
              for (int k = ki + 2; k <= j - 1; ++k) {
                d1 += Tm(k, j) * w1[k - 1];
                d2 += Tm(k, j) * w2[k - 1];
              }
              w1[j - 1] -= d1;
              w2[j - 1] -= d2;
              const T ablk = Tm(j, j);
              T b[1 * 2];
              b[0] = w1[j - 1];
              b[1] = w2[j - 1];
              T x[2 * 2];
              T scale, xnorm;
              laln2_solve<T>(true, 1, 2, smin, one, &ablk, 1, one, one, b, 1, wr, -wi, x, 2, &scale,
                             &xnorm);
              if (scale != one) {
                for (int k = ki; k <= n; ++k) {
                  w1[k - 1] *= scale;
                  w2[k - 1] *= scale;
                }
              }
              w1[j - 1] = x[0];
              w2[j - 1] = x[2];
              vmax = dmax(dmax(dabs(w1[j - 1]), dabs(w2[j - 1])), vmax);
              vcrit = bignum / vmax;
            } else {
              const T beta = dmax(dmax(w1[j - 1], w1[j]), dmax(w2[j - 1], w2[j]));
              if (beta > vcrit) {
                const T rec = one / vmax;
                for (int k = ki; k <= n; ++k) {
                  w1[k - 1] *= rec;
                  w2[k - 1] *= rec;
                }
                vmax = one;
                vcrit = bignum;
              }
              T d1r = zero, d1i = zero, d2r = zero, d2i = zero;
              for (int k = ki + 2; k <= j - 1; ++k) {
                d1r += Tm(k, j) * w1[k - 1];
                d1i += Tm(k, j) * w2[k - 1];
                d2r += Tm(k, j + 1) * w1[k - 1];
                d2i += Tm(k, j + 1) * w2[k - 1];
              }
              w1[j - 1] -= d1r;
              w2[j - 1] -= d1i;
              w1[j] -= d2r;
              w2[j] -= d2i;
              T ablk[2 * 2];
              ablk[0] = Tm(j, j);
              ablk[1] = Tm(j + 1, j);
              ablk[2] = Tm(j, j + 1);
              ablk[3] = Tm(j + 1, j + 1);
              T b[2 * 2];
              b[0] = w1[j - 1];
              b[1] = w1[j];
              b[2] = w2[j - 1];
              b[3] = w2[j];
              T x[2 * 2];
              T scale, xnorm;
              laln2_solve<T>(true, 2, 2, smin, one, ablk, 2, one, one, b, 2, wr, -wi, x, 2, &scale,
                             &xnorm);
              if (scale != one) {
                for (int k = ki; k <= n; ++k) {
                  w1[k - 1] *= scale;
                  w2[k - 1] *= scale;
                }
              }
              w1[j - 1] = x[0];
              w1[j] = x[1];
              w2[j - 1] = x[2];
              w2[j] = x[3];
              vmax = dmax(dmax(dmax(dabs(w1[j - 1]), dabs(w1[j])),
                               dmax(dabs(w2[j - 1]), dabs(w2[j]))),
                          vmax);
              vcrit = bignum / vmax;
            }
          }
          T emax = zero;
          for (int k = ki; k <= n; ++k) {
            emax = dmax(emax, dabs(w1[k - 1]) + dabs(w2[k - 1]));
          }
          const T remax = one / emax;
          for (int k = ki; k <= n; ++k) {
            vl_[(k - 1) + (is - 1) * ldvl_] = remax * w1[k - 1];
            vl_[(k - 1) + is * ldvl_] = remax * w2[k - 1];
          }
          for (int k = 1; k <= ki - 1; ++k) {
            vl_[(k - 1) + (is - 1) * ldvl_] = zero;
            vl_[(k - 1) + is * ldvl_] = zero;
          }
          is += 2;
        }
      }
    }
  }
};

} // namespace

template<typename T>
void trevc3(const wwr::wwrStream_t stream, const bool want_left, const bool want_right, const int n,
            const T *const t, const int ldt, T *const vl, const int ldvl, T *const vr,
            const int ldvr, T *const work) {
  const Trevc3Functor<T> functor{want_left, want_right, n, t, ldt, vl, ldvl, vr, ldvr, work};
  wwr::extension::parallel_for<std::size_t>(stream, std::size_t{1}, functor);
}

// One per supported precision, matching trevc3_bridge.h's declaration and
// interface.cppm's extern-template list -- a type added here without being added
// there, or vice versa, links against nothing.
template void trevc3<float>(wwr::wwrStream_t, bool, bool, int, const float *, int, float *, int,
                            float *, int, float *);
template void trevc3<double>(wwr::wwrStream_t, bool, bool, int, const double *, int, double *, int,
                             double *, int, double *);

} // namespace calaman::device
