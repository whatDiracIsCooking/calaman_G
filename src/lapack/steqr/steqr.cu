// steqr.cu
//
// The device half of calaman.steqr: reference ?steqr (LAPACK 3.12.0) in ONE
// block. Thread 0 runs the whole implicit QL/QR chase on (d, e) -- split
// search, ?lanst/?lascl scaling, the 2x2 ?lae2/?laev2 shortcut, the shifted
// sweep with ?lartg -- recording each sweep's rotations in work, line for line
// with dsteqr.f (1-based D/E/W accessors). Whenever the reference touches Z
// (DLASR after a sweep, DSWAP in the closing selection sort) thread 0 pauses and
// publishes a command in shared memory; the whole block then applies it
// (lasr_block, or a column swap) and thread 0 resumes. The chase is therefore a
// resumable state machine (Label = the reference's GO TO targets).
//
// The machine constants (DLAMCH 'E'/'S' and the ssfmin/ssfmax scaling bounds)
// and ?lartg's thresholds are computed on the host and passed in. COMPZ = N
// sorts by the same selection sort with no swaps instead of DLASRT: the sorted
// eigenvalues are identical. Shared unchanged between both backends.
#include "steqr_bridge.h"

#include "lapack/lanst/lanst.h"
#include "lapack/lartg/lartg.cuh"
#include "lapack/lasr/lasr.h"
#include "lapack/sym2x2/sym2x2.cuh"

#include <wrappers/math/math.cuh>

#include <cmath>
#include <cstddef>
#include <limits>

namespace calaman::device {

namespace {

constexpr unsigned int kBlock = 256;
constexpr int kMaxIt = 30; // MAXIT: iterations allowed per eigenvalue

/// @brief dsteqr's machine constants, built on the host
template<typename T>
struct SteqrConsts {
  T eps;    ///< DLAMCH('E')
  T eps2;   ///< eps^2
  T safmin; ///< DLAMCH('S')
  T ssfmax; ///< sqrt(1/safmin) / 3
  T ssfmin; ///< sqrt(safmin) / eps2
};

template<typename T>
SteqrConsts<T> steqr_consts() {
  const T eps = std::numeric_limits<T>::epsilon() / T{2};
  const T eps2 = eps * eps;
  const T safmin = std::numeric_limits<T>::min();
  const T safmax = T{1} / safmin;
  return {eps, eps2, safmin, std::sqrt(safmax) / T{3}, std::sqrt(safmin) / eps2};
}

/// What thread 0 asks the block to do to Z next.
enum class Op : int { done, lasr, swap };

/// A block command. Plain aggregate (no initializers) so it can live in __shared__.
struct Cmd {
  Op op;
  Direct direct; ///< lasr: Direct::B (QL) or Direct::F (QR)
  int mm;        ///< lasr: columns of Z rotated
  int col;       ///< lasr: first Z column / work offset (0-based); swap: column i
  int other;     ///< swap: column k (0-based)
};

/// The reference's GO TO targets: 10 split, 40 QL, 90 QR, 140 unscale, 160 sort.
enum class Label : int { split, ql, qr, unscale, sort };

/// Thread 0's state between commands (1-based indices, as dsteqr.f).
template<typename T>
struct Chase {
  Label label;
  int l1, l, lend, lsv, lendsv, iscale, jtot, nmaxit, ii;
  T anorm;
};

/// DLASCL('G', 0, 0, cfrom, cto, len, 1, x, ...): x <- x * cto / cfrom without
/// over/underflow, by the reference's multiplier chain. One thread.
template<typename T>
__device__ void lascl_vec(const T cfrom, const T cto, const int len, T *const x, const T smlnum) {
  const T bignum = T{1} / smlnum;
  T cfromc = cfrom;
  T ctoc = cto;
  bool done = false;
  while (!done) {
    const T cfrom1 = cfromc * smlnum;
    T mul;
    if (cfrom1 == cfromc) { // cfromc is inf: a correctly signed zero, or NaN
      mul = ctoc / cfromc;
      done = true;
    } else {
      const T cto1 = ctoc / bignum;
      if (cto1 == ctoc) { // ctoc is 0 or inf
        mul = ctoc;
        done = true;
        cfromc = T{1};
      } else if (wwr::fabs(cfrom1) > wwr::fabs(ctoc) && ctoc != T{0}) {
        mul = smlnum;
        cfromc = cfrom1;
      } else if (wwr::fabs(cto1) > wwr::fabs(cfromc)) {
        mul = bignum;
        ctoc = cto1;
      } else {
        mul = ctoc / cfromc;
        done = true;
        if (mul == T{1}) {
          return;
        }
      }
    }
    for (int i = 0; i < len; ++i) {
      x[i] *= mul;
    }
  }
}

/// @brief Run thread 0's chase until the next Z command (or the end)
template<typename T>
__device__ Cmd advance(Chase<T> &st, const SteqrConsts<T> &k, const LartgThresholds<T> &th,
                       const bool wantz, const int n, T *const d, T *const e, T *const work,
                       int *const info) {
  auto D = [=](int i) -> T & { return d[i - 1]; };
  auto E = [=](int i) -> T & { return e[i - 1]; };
  auto W = [=](int i) -> T & { return work[i - 1]; };
  for (;;) {
    switch (st.label) {
    case Label::split: { // 10
      if (st.l1 > n) {
        st.label = Label::sort;
        st.ii = 2;
        break;
      }
      if (st.l1 > 1) {
        E(st.l1 - 1) = T{0};
      }
      int m = n;
      for (int mm = st.l1; mm <= n - 1; ++mm) {
        const T tst = wwr::fabs(E(mm));
        if (tst == T{0}) {
          m = mm;
          break;
        }
        if (tst <= (wwr::sqrt(wwr::fabs(D(mm))) * wwr::sqrt(wwr::fabs(D(mm + 1)))) * k.eps) {
          E(mm) = T{0};
          m = mm;
          break;
        }
      }
      st.l = st.l1;
      st.lsv = st.l;
      st.lend = m;
      st.lendsv = st.lend;
      st.l1 = m + 1;
      if (st.lend == st.l) {
        break; // label stays split
      }
      // Scale the submatrix in rows and columns l to lend.
      const int len = st.lend - st.l + 1;
      st.anorm = lanst_max_abs(static_cast<std::size_t>(len), &D(st.l), &E(st.l));
      st.iscale = 0;
      if (st.anorm == T{0}) {
        break;
      }
      if (st.anorm > k.ssfmax) {
        st.iscale = 1;
        lascl_vec(st.anorm, k.ssfmax, len, &D(st.l), k.safmin);
        lascl_vec(st.anorm, k.ssfmax, len - 1, &E(st.l), k.safmin);
      } else if (st.anorm < k.ssfmin) {
        st.iscale = 2;
        lascl_vec(st.anorm, k.ssfmin, len, &D(st.l), k.safmin);
        lascl_vec(st.anorm, k.ssfmin, len - 1, &E(st.l), k.safmin);
      }
      // QL when the bottom end is the larger, else QR.
      if (wwr::fabs(D(st.lend)) < wwr::fabs(D(st.l))) {
        st.lend = st.lsv;
        st.l = st.lendsv;
      }
      st.label = st.lend > st.l ? Label::ql : Label::qr;
      break;
    }

    case Label::ql: { // 40: look for a small subdiagonal element
      const int l = st.l;
      const int lend = st.lend;
      int m = lend;
      for (int mm = l; mm <= lend - 1; ++mm) {
        const T tst = wwr::fabs(E(mm)) * wwr::fabs(E(mm));
        if (tst <= (k.eps2 * wwr::fabs(D(mm))) * wwr::fabs(D(mm + 1)) + k.safmin) {
          m = mm;
          break;
        }
      }
      if (m < lend) {
        E(m) = T{0};
      }
      T p = D(l);
      if (m == l) { // 80: eigenvalue found
        D(l) = p;
        st.l = l + 1;
        st.label = st.l <= lend ? Label::ql : Label::unscale;
        break;
      }
      if (m == l + 1) { // 2x2 block
        T rt1, rt2;
        if (wantz) {
          T c, s;
          laev2_scalar(D(l), E(l), D(l + 1), &rt1, &rt2, &c, &s);
          W(l) = c;
          W(n - 1 + l) = s;
        } else {
          lae2_scalar(D(l), E(l), D(l + 1), &rt1, &rt2);
        }
        D(l) = rt1;
        D(l + 1) = rt2;
        E(l) = T{0};
        st.l = l + 2;
        st.label = st.l <= lend ? Label::ql : Label::unscale;
        if (wantz) {
          return Cmd{Op::lasr, Direct::B, 2, l - 1, 0};
        }
        break;
      }
      if (st.jtot == st.nmaxit) {
        st.label = Label::unscale;
        break;
      }
      ++st.jtot;
      // Form the shift.
      T g = (D(l + 1) - p) / (T{2} * E(l));
      T r = lapy2_scalar(g, T{1});
      g = D(m) - p + (E(l) / (g + wwr::copysign(r, g)));
      T s = T{1};
      T c = T{1};
      p = T{0};
      for (int i = m - 1; i >= l; --i) {
        const T f = s * E(i);
        const T b = c * E(i);
        lartg_scalar(g, f, th, &c, &s, &r);
        if (i != m - 1) {
          E(i + 1) = r;
        }
        g = D(i + 1) - p;
        r = (D(i) - g) * s + T{2} * c * b;
        p = s * r;
        D(i + 1) = g + p;
        g = c * r - b;
        if (wantz) {
          W(i) = c;
          W(n - 1 + i) = -s;
        }
      }
      D(l) = D(l) - p;
      E(l) = g;
      // label stays ql
      if (wantz) {
        return Cmd{Op::lasr, Direct::B, m - l + 1, l - 1, 0};
      }
      break;
    }

    case Label::qr: { // 90: look for a small superdiagonal element
      const int l = st.l;
      const int lend = st.lend;
      int m = lend;
      for (int mm = l; mm >= lend + 1; --mm) {
        const T tst = wwr::fabs(E(mm - 1)) * wwr::fabs(E(mm - 1));
        if (tst <= (k.eps2 * wwr::fabs(D(mm))) * wwr::fabs(D(mm - 1)) + k.safmin) {
          m = mm;
          break;
        }
      }
      if (m > lend) {
        E(m - 1) = T{0};
      }
      T p = D(l);
      if (m == l) { // 130: eigenvalue found
        D(l) = p;
        st.l = l - 1;
        st.label = st.l >= lend ? Label::qr : Label::unscale;
        break;
      }
      if (m == l - 1) { // 2x2 block
        T rt1, rt2;
        if (wantz) {
          T c, s;
          laev2_scalar(D(l - 1), E(l - 1), D(l), &rt1, &rt2, &c, &s);
          W(m) = c;
          W(n - 1 + m) = s;
        } else {
          lae2_scalar(D(l - 1), E(l - 1), D(l), &rt1, &rt2);
        }
        D(l - 1) = rt1;
        D(l) = rt2;
        E(l - 1) = T{0};
        st.l = l - 2;
        st.label = st.l >= lend ? Label::qr : Label::unscale;
        if (wantz) {
          return Cmd{Op::lasr, Direct::F, 2, m - 1, 0};
        }
        break;
      }
      if (st.jtot == st.nmaxit) {
        st.label = Label::unscale;
        break;
      }
      ++st.jtot;
      // Form the shift.
      T g = (D(l - 1) - p) / (T{2} * E(l - 1));
      T r = lapy2_scalar(g, T{1});
      g = D(m) - p + (E(l - 1) / (g + wwr::copysign(r, g)));
      T s = T{1};
      T c = T{1};
      p = T{0};
      for (int i = m; i <= l - 1; ++i) {
        const T f = s * E(i);
        const T b = c * E(i);
        lartg_scalar(g, f, th, &c, &s, &r);
        if (i != m) {
          E(i - 1) = r;
        }
        g = D(i) - p;
        r = (D(i + 1) - g) * s + T{2} * c * b;
        p = s * r;
        D(i) = g + p;
        g = c * r - b;
        if (wantz) {
          W(i) = c;
          W(n - 1 + i) = s;
        }
      }
      D(l) = D(l) - p;
      E(l - 1) = g;
      // label stays qr
      if (wantz) {
        return Cmd{Op::lasr, Direct::F, l - m + 1, m - 1, 0};
      }
      break;
    }

    case Label::unscale: { // 140
      const int len = st.lendsv - st.lsv + 1;
      if (st.iscale == 1) {
        lascl_vec(k.ssfmax, st.anorm, len, &D(st.lsv), k.safmin);
        lascl_vec(k.ssfmax, st.anorm, len - 1, &E(st.lsv), k.safmin);
      } else if (st.iscale == 2) {
        lascl_vec(k.ssfmin, st.anorm, len, &D(st.lsv), k.safmin);
        lascl_vec(k.ssfmin, st.anorm, len - 1, &E(st.lsv), k.safmin);
      }
      if (st.jtot < st.nmaxit) {
        st.label = Label::split;
        break;
      }
      // No convergence: INFO counts the off-diagonals left non-zero.
      int count = 0;
      for (int i = 1; i <= n - 1; ++i) {
        if (E(i) != T{0}) {
          ++count;
        }
      }
      *info = count;
      return Cmd{Op::done, Direct::F, 0, 0, 0};
    }

    case Label::sort: { // 160: selection sort, minimising swaps of Z columns
      while (st.ii <= n) {
        const int i = st.ii - 1;
        int kk = i;
        T p = D(i);
        for (int j = st.ii; j <= n; ++j) {
          if (D(j) < p) {
            kk = j;
            p = D(j);
          }
        }
        ++st.ii;
        if (kk != i) {
          D(kk) = D(i);
          D(i) = p;
          if (wantz) {
            return Cmd{Op::swap, Direct::F, 0, i - 1, kk - 1};
          }
        }
      }
      return Cmd{Op::done, Direct::F, 0, 0, 0};
    }
    }
  }
}

/// @brief [kernel] ?steqr on (d, e), Z updated by the whole block
template<typename T>
__global__ void steqr_kernel(const CompZ compz, const int n, T *const d, T *const e, T *const z,
                             const int ldz, T *const work, int *const info,
                             const SteqrConsts<T> consts, const LartgThresholds<T> th) {
  const unsigned int tid = threadIdx.x;
  const bool wantz = compz != CompZ::N;
  const std::size_t ld = static_cast<std::size_t>(ldz);
  const std::size_t nn = static_cast<std::size_t>(n);

  if (tid == 0) {
    *info = 0;
  }
  if (compz == CompZ::I) { // DLASET('Full', N, N, ZERO, ONE, Z, LDZ)
    for (std::size_t idx = tid; idx < nn * nn; idx += blockDim.x) {
      const std::size_t i = idx % nn;
      const std::size_t j = idx / nn;
      z[i + j * ld] = i == j ? T{1} : T{0};
    }
  }
  if (n <= 1) {
    return;
  }

  __shared__ Cmd cmd;
  Chase<T> st{};
  st.label = Label::split;
  st.l1 = 1;
  st.nmaxit = n * kMaxIt;
  for (;;) {
    if (tid == 0) {
      cmd = advance(st, consts, th, wantz, n, d, e, work, info);
    }
    __syncthreads();
    const Cmd c = cmd;
    if (c.op == Op::done) {
      return;
    }
    if (c.op == Op::lasr) {
      // Opens with a barrier (every thread has read cmd) and closes with one.
      lasr_block(Side::R, Pivot::V, c.direct, nn, static_cast<std::size_t>(c.mm), work + c.col,
                 work + (n - 1) + c.col, z + static_cast<std::size_t>(c.col) * ld, ld);
    } else {
      T *const zi = z + static_cast<std::size_t>(c.col) * ld;
      T *const zk = z + static_cast<std::size_t>(c.other) * ld;
      for (std::size_t r = tid; r < nn; r += blockDim.x) {
        const T t = zi[r];
        zi[r] = zk[r];
        zk[r] = t;
      }
      __syncthreads();
    }
  }
}

} // namespace

template<typename T>
void steqr(const wwr::wwrStream_t stream, const CompZ compz, const int n, T *const d, T *const e,
           T *const z, const int ldz, T *const work, int *const info) {
  steqr_kernel<T><<<1, kBlock, 0, stream>>>(compz, n, d, e, z, ldz, work, info,
                                            steqr_consts<T>(), lartg_thresholds<T>());
}

// One per supported type, matching steqr_bridge.h and interface.cppm's
// `extern template` list -- float and double.
template void steqr<float>(wwr::wwrStream_t, CompZ, int, float *, float *, float *, int, float *,
                           int *);
template void steqr<double>(wwr::wwrStream_t, CompZ, int, double *, double *, double *, int,
                            double *, int *);

} // namespace calaman::device
