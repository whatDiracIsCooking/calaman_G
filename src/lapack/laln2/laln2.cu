// laln2.cu
//
// The device-kernel half of calaman.laln2: a single-thread port of LAPACK's
// ?laln2, launched through wwr.extension.parallel_for over a grid of one. The
// whole routine is O(1) -- it solves (ca*op(A) - w*D) X = SCALE*B for a system
// of order at most 2 with a real or complex right-hand side -- so there is
// nothing to parallelise; one thread runs the entire reference algorithm, and
// the module wrapper is a thin front door.
//
// FAITHFUL TO THE REFERENCE, by design: the test oracle is reference ?laln2, so
// the arithmetic, the complete-pivoting order, the SMINI perturbation and the
// scale/xnorm formulas all mirror dlaln2.f line for line. The complex divisions
// (the NW==2 cases) go through ladiv_scalar from "lapack/ladiv/ladiv.h" -- the SAME
// host/device Smith's-algorithm helper the reference reaches as DLADIV, included
// here so the call has device linkage (issue #94 depends on #86's ?ladiv).
//
// LAPACK's DLAMCH('S') (the safe minimum) is FLT_MIN / DBL_MIN -- the <cfloat>
// macros, which are plain constants usable on the device, where numeric_limits
// is not (its members are __host__ constexpr, rejected by nvcc without
// --expt-relaxed-constexpr). Indices track the Fortran 1-based scheme; CRV/CIV
// are the column-major flattenings of the 2x2 CR/CI, and IPIVOT is the Fortran
// DATA array read column-major.
//
// REAL ELEMENT TYPE: the reference is SLALN2 / DLALN2 (the scalar w may be
// complex, but the matrix A and the arithmetic are real), so the surface is
// float / double, matching it. Shared unchanged between both backends: a .cu is
// compiled by the backend compiler, and parallel_for's launch machinery arrives
// through its device header.
#include "laln2_bridge.h"

#include "lapack/ladiv/ladiv.h"

#include "extension/parallel_for/parallel_for.cuh"

#include <cfloat>
#include <cstddef>

namespace calaman::device {

namespace {

// abs / max in exact arithmetic (no rounding), so they match LAPACK's intrinsic
// ABS / MAX bit for bit -- the control flow (pivot choice, SMINI tests) must.
template<typename T>
__device__ __forceinline__ T dabs(const T v) {
  return v < T{0} ? -v : v;
}
template<typename T>
__device__ __forceinline__ T dmax(const T a, const T b) {
  return a > b ? a : b;
}

// LAPACK's DLAMCH('S'), device-side: the safe minimum. numeric_limits is a
// __host__ constexpr the device pass rejects, so this reads the <cfloat> macro.
template<typename T>
struct lamch;
template<>
struct lamch<float> {
  static __device__ __forceinline__ float sfmin() { return FLT_MIN; }
};
template<>
struct lamch<double> {
  static __device__ __forceinline__ double sfmin() { return DBL_MIN; }
};

// The per-call solve. Members are const scalar / pointer, as device_functor
// requires: trivially copyable, and the const deletes the copy-assignment the
// grid-constant kernel copy would otherwise permit. One thread runs operator();
// the index is ignored (the grid is a single element).
template<typename T>
struct Laln2Functor {
  const bool ltrans_;
  const int na_;
  const int nw_;
  const T smin_;
  const T ca_;
  const T *const a_;
  const int lda_;
  const T d1_;
  const T d2_;
  const T *const b_;
  const int ldb_;
  const T wr_;
  const T wi_;
  T *const x_;
  const int ldx_;
  T *const scale_;
  T *const xnorm_;
  int *const info_;

  __device__ void operator()(std::size_t) const {
    const T zero = T{0};
    const T one = T{1};
    const T two = T{2};

    const T smlnum = two * lamch<T>::sfmin(); // DLAMCH('S') scaled, as DLALN2
    const T bignum = one / smlnum;
    const T smini = dmax(smin_, smlnum);

    // Column-major, 1-based accessors on A / B; X is written through setX.
    const T *const a = a_;
    const int lda = lda_;
    const T *const b = b_;
    const int ldb = ldb_;
    T *const x = x_;
    const int ldx = ldx_;
    auto A = [=](int i, int j) -> T { return a[(i - 1) + (j - 1) * lda]; };
    auto B = [=](int i, int j) -> T { return b[(i - 1) + (j - 1) * ldb]; };
    auto setX = [=](int i, int j, T v) { x[(i - 1) + (j - 1) * ldx] = v; };

    int info = 0;
    T scale = one;

    // RSWAP / ZSWAP / IPIVOT -- the Fortran DATA arrays, 1-based (index 0 unused).
    // IPIVOT is read column-major: IPIVOT(row, col), col == ICMAX.
    const bool rswap[5] = {false, false, true, false, true};
    const bool zswap[5] = {false, false, false, true, true};
    const int ipivot[5][5] = {
        {0, 0, 0, 0, 0}, {0, 1, 2, 3, 4}, {0, 2, 1, 4, 3}, {0, 3, 4, 1, 2}, {0, 4, 3, 2, 1}};

    if (na_ == 1) {
      if (nw_ == 1) {
        // Real 1x1 system   C X = B.
        T csr = ca_ * A(1, 1) - wr_ * d1_;
        T cnorm = dabs(csr);
        if (cnorm < smini) {
          csr = smini;
          cnorm = smini;
          info = 1;
        }
        const T bnorm = dabs(B(1, 1));
        if (cnorm < one && bnorm > one) {
          if (bnorm > bignum * cnorm) {
            scale = one / bnorm;
          }
        }
        const T xr = (B(1, 1) * scale) / csr;
        setX(1, 1, xr);
        *xnorm_ = dabs(xr);
      } else {
        // Complex 1x1 system (w is complex).
        T csr = ca_ * A(1, 1) - wr_ * d1_;
        T csi = -wi_ * d1_;
        T cnorm = dabs(csr) + dabs(csi);
        if (cnorm < smini) {
          csr = smini;
          csi = zero;
          cnorm = smini;
          info = 1;
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
        *xnorm_ = dabs(xr) + dabs(xi);
      }
      *scale_ = scale;
      *info_ = info;
      return;
    }

    // 2x2 system. Build the real part of C = ca*op(A) - w*D as a column-major
    // CR(2,2), flattened to crv[1..4] = {CR11, CR21, CR12, CR22}.
    T crv[5];
    crv[1] = ca_ * A(1, 1) - wr_ * d1_; // CR(1,1)
    crv[4] = ca_ * A(2, 2) - wr_ * d2_; // CR(2,2)
    if (ltrans_) {
      crv[3] = ca_ * A(2, 1); // CR(1,2)
      crv[2] = ca_ * A(1, 2); // CR(2,1)
    } else {
      crv[2] = ca_ * A(2, 1); // CR(2,1)
      crv[3] = ca_ * A(1, 2); // CR(1,2)
    }

    if (nw_ == 1) {
      // Real 2x2 system (w is real). Find the largest element in C.
      T cmax = zero;
      int icmax = 0;
      for (int j = 1; j <= 4; ++j) {
        if (dabs(crv[j]) > cmax) {
          cmax = dabs(crv[j]);
          icmax = j;
        }
      }

      // If norm(C) < SMINI, use SMINI*identity.
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
        *xnorm_ = temp * bnorm;
        *scale_ = scale;
        *info_ = 1;
        return;
      }

      // Gaussian elimination with complete pivoting.
      const T ur11 = crv[icmax];
      const T cr21 = crv[ipivot[2][icmax]];
      const T ur12 = crv[ipivot[3][icmax]];
      const T cr22 = crv[ipivot[4][icmax]];
      const T ur11r = one / ur11;
      const T lr21 = ur11r * cr21;
      T ur22 = cr22 - ur12 * lr21;
      if (dabs(ur22) < smini) {
        ur22 = smini;
        info = 1;
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

      // Further scaling if norm(A) norm(X) > overflow.
      if (xnorm > one && cmax > one) {
        if (xnorm > bignum / cmax) {
          const T temp = cmax / bignum;
          setX(1, 1, temp * (zswap[icmax] ? xr2 : xr1));
          setX(2, 1, temp * (zswap[icmax] ? xr1 : xr2));
          xnorm = temp * xnorm;
          scale = temp * scale;
        }
      }
      *xnorm_ = xnorm;
      *scale_ = scale;
      *info_ = info;
      return;
    }

    // Complex 2x2 system (w is complex). Build the imaginary part CI(2,2)
    // flattened to civ[1..4] = {CI11, CI21, CI12, CI22}.
    T civ[5];
    civ[1] = -wi_ * d1_; // CI(1,1)
    civ[2] = zero;       // CI(2,1)
    civ[3] = zero;       // CI(1,2)
    civ[4] = -wi_ * d2_; // CI(2,2)

    T cmax = zero;
    int icmax = 0;
    for (int j = 1; j <= 4; ++j) {
      if (dabs(crv[j]) + dabs(civ[j]) > cmax) {
        cmax = dabs(crv[j]) + dabs(civ[j]);
        icmax = j;
      }
    }

    // If norm(C) < SMINI, use SMINI*identity.
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
      *xnorm_ = temp * bnorm;
      *scale_ = scale;
      *info_ = 1;
      return;
    }

    // Gaussian elimination with complete pivoting.
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
      // Off-diagonals of pivoted C are real.
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
      // Diagonals of pivoted C are real.
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
      info = 1;
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

    // Further scaling if norm(A) norm(X) > overflow.
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
    *xnorm_ = xnorm;
    *scale_ = scale;
    *info_ = info;
  }
};

} // namespace

template<typename T>
void laln2(const wwr::wwrStream_t stream, const bool ltrans, const int na, const int nw,
           const T smin, const T ca, const T *const a, const int lda, const T d1, const T d2,
           const T *const b, const int ldb, const T wr, const T wi, T *const x, const int ldx,
           T *const scale, T *const xnorm, int *const info) {
  const Laln2Functor<T> functor{ltrans, na, nw, smin, ca, a,  lda,   d1,    d2,
                                b,      ldb, wr, wi,   x,  ldx, scale, xnorm, info};
  wwr::extension::parallel_for<std::size_t>(stream, std::size_t{1}, functor);
}

// One per supported precision, matching laln2_bridge.h's declaration and
// interface.cppm's extern-template list -- a type added here without being added
// there, or vice versa, links against nothing.
template void laln2<float>(wwr::wwrStream_t, bool, int, int, float, float, const float *, int, float,
                           float, const float *, int, float, float, float *, int, float *, float *,
                           int *);
template void laln2<double>(wwr::wwrStream_t, bool, int, int, double, double, const double *, int,
                            double, double, const double *, int, double, double, double *, int,
                            double *, double *, int *);

} // namespace calaman::device
