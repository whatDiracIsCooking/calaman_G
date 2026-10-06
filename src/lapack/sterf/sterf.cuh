/**
 * @file sterf.cuh
 * @brief sterf_serial: LAPACK's ?sterf, run by one calling device thread
 *
 * Eigenvalues of the symmetric tridiagonal (d, e) by the root-free
 * Pal-Walker-Kahan QL/QR iteration, ported from LAPACK 3.12.0's dsterf so the
 * branch choices match the reference: each unreduced block is scaled in place
 * (?lascl 'G' semantics) when its ?lanst('M') norm leaves [ssfmin, ssfmax],
 * and the converged d is sorted ascending with lasrt_serial. On
 * non-convergence (30*n QL/QR sweeps in total) d is left unsorted, exactly as
 * the reference leaves it. Real only (float, double); no workspace.
 *
 * calaman.sterf's kernel is one call of this; another kernel (?syev) can call
 * it in-thread. Reach it as "lapack/sterf/sterf.cuh" by linking the INTERFACE
 * target calaman::sterf::header, which carries the src/ root plus sym2x2,
 * lanst::header and lasrt::header. Device-only: include it from a .cu.
 *
 * Usage:
 *   #include "lapack/sterf/sterf.cuh"
 *   const int info = calaman::device::sterf_serial(n, d, e);
 */

#pragma once

#include "lapack/lanst/lanst.h"
#include "lapack/lasrt/lasrt.cuh"
#include "lapack/sym2x2/sym2x2.cuh"

#include <wrappers/math/math.cuh>

#include <cfloat>
#include <cstddef>
#include <type_traits>

namespace calaman::device {

namespace sterf_detail {

/// DLAMCH('E') (eps, half of FLT/DBL_EPSILON under rounding) and ('S').
template<typename T>
__device__ __forceinline__ T lamch_e() {
  if constexpr (std::is_same_v<T, float>) {
    return FLT_EPSILON * 0.5f;
  } else {
    return DBL_EPSILON * 0.5;
  }
}
template<typename T>
__device__ __forceinline__ T lamch_s() {
  if constexpr (std::is_same_v<T, float>) {
    return FLT_MIN;
  } else {
    return DBL_MIN;
  }
}

/// @brief ?lascl('G') of the contiguous length-@p m vector @p x: x *= cto/cfrom
///
/// The reference's overflow-safe multiplier chain, applied in-thread; returns
/// early once a step's multiplier is exactly one, as DLASCL does.
template<typename T>
__device__ void lascl_g_serial(const T cfrom, const T cto, const int m, T *const x) {
  if (m <= 0) {
    return;
  }
  const T smlnum = lamch_s<T>();
  const T bignum = T{1} / smlnum;
  T cfromc = cfrom;
  T ctoc = cto;
  bool done = false;
  while (!done) {
    const T cfrom1 = cfromc * smlnum;
    T mul;
    if (cfrom1 == cfromc) {
      // cfromc is an inf: a signed zero for finite ctoc, NaN for infinite.
      mul = ctoc / cfromc;
      done = true;
    } else {
      const T cto1 = ctoc / bignum;
      if (cto1 == ctoc) {
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
    for (int i = 0; i < m; ++i) {
      x[i] *= mul;
    }
  }
}

} // namespace sterf_detail

/// @brief Eigenvalues of the order-@p n symmetric tridiagonal (d, e) (?sterf)
///
/// On return @p d holds the eigenvalues ascending and @p e is destroyed. Returns
/// INFO: 0 on success, else the count of e entries not yet zero when the 30*n
/// iteration budget ran out (d then unsorted). @p e is unread when n <= 1.
///
/// @tparam T Element type (float, double)
template<typename T>
__device__ int sterf_serial(const int n, T *const d, T *const e) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "sterf_serial is real-only (float, double)");
  using sterf_detail::lascl_g_serial;
  constexpr int kMaxIt = 30;
  const T zero{0}, one{1}, two{2}, three{3};

  if (n <= 1) {
    return 0;
  }
  // 1-based accessors, so the body tracks dsterf.f line for line.
  auto D = [d](int i) -> T & { return d[i - 1]; };
  auto E = [e](int i) -> T & { return e[i - 1]; };

  const T eps = sterf_detail::lamch_e<T>();
  const T eps2 = eps * eps;
  const T safmin = sterf_detail::lamch_s<T>();
  const T safmax = one / safmin;
  const T ssfmax = wwr::sqrt(safmax) / three;
  const T ssfmin = wwr::sqrt(safmin) / eps2;

  const int nmaxit = n * kMaxIt;
  int jtot = 0;
  int info = 0;

  // Determine where the matrix splits and choose QL or QR for each block.
  int l1 = 1;
  while (l1 <= n) {
    if (l1 > 1) {
      E(l1 - 1) = zero;
    }
    int m = l1;
    for (; m <= n - 1; ++m) {
      if (wwr::fabs(E(m)) <= (wwr::sqrt(wwr::fabs(D(m))) * wwr::sqrt(wwr::fabs(D(m + 1)))) * eps) {
        E(m) = zero;
        break;
      }
    }
    // Fell through: m == n.
    int l = l1;
    const int lsv = l;
    int lend = m;
    const int lendsv = lend;
    l1 = m + 1;
    if (lend == l) {
      continue;
    }

    // Scale the block in rows and columns l..lend.
    const T anorm = lanst_max_abs(static_cast<std::size_t>(lend - l + 1), &D(l), &E(l));
    int iscale = 0;
    if (anorm == zero) {
      continue;
    }
    if (anorm > ssfmax) {
      iscale = 1;
      lascl_g_serial(anorm, ssfmax, lend - l + 1, &D(l));
      lascl_g_serial(anorm, ssfmax, lend - l, &E(l));
    } else if (anorm < ssfmin) {
      iscale = 2;
      lascl_g_serial(anorm, ssfmin, lend - l + 1, &D(l));
      lascl_g_serial(anorm, ssfmin, lend - l, &E(l));
    }

    for (int i = l; i <= lend - 1; ++i) {
      E(i) = E(i) * E(i);
    }

    // Choose between QL and QR iteration.
    if (wwr::fabs(D(lend)) < wwr::fabs(D(l))) {
      lend = lsv;
      l = lendsv;
    }

    if (lend >= l) {
      // QL iteration: look for a small subdiagonal element.
      while (true) {
        int mm = lend;
        if (l != lend) {
          for (mm = l; mm <= lend - 1; ++mm) {
            if (wwr::fabs(E(mm)) <= eps2 * wwr::fabs(D(mm) * D(mm + 1))) {
              break;
            }
          }
        }
        if (mm < lend) {
          E(mm) = zero;
        }
        T p = D(l);
        if (mm == l) {
          // Eigenvalue found.
          D(l) = p;
          ++l;
          if (l <= lend) {
            continue;
          }
          break;
        }
        // A remaining 2x2 block: ?lae2 gives its eigenvalues.
        if (mm == l + 1) {
          const T rte = wwr::sqrt(E(l));
          T rt1, rt2;
          lae2_scalar(D(l), rte, D(l + 1), &rt1, &rt2);
          D(l) = rt1;
          D(l + 1) = rt2;
          E(l) = zero;
          l += 2;
          if (l <= lend) {
            continue;
          }
          break;
        }
        if (jtot == nmaxit) {
          break;
        }
        ++jtot;

        // Form the shift.
        T rte = wwr::sqrt(E(l));
        T sigma = (D(l + 1) - p) / (two * rte);
        T r = lapy2_scalar(sigma, one);
        sigma = p - (rte / (sigma + wwr::copysign(r, sigma)));

        T c = one;
        T s = zero;
        T gamma = D(mm) - sigma;
        p = gamma * gamma;

        // Inner loop.
        const int mm1 = mm - 1;
        for (int i = mm1; i >= l; --i) {
          const T bb = E(i);
          r = p + bb;
          if (i != mm1) {
            E(i + 1) = s * r;
          }
          const T oldc = c;
          c = p / r;
          s = bb / r;
          const T oldgam = gamma;
          const T alpha = D(i);
          gamma = c * (alpha - sigma) - s * oldgam;
          D(i + 1) = oldgam + (alpha - gamma);
          if (c != zero) {
            p = (gamma * gamma) / c;
          } else {
            p = oldc * bb;
          }
        }
        E(l) = s * p;
        D(l) = sigma + gamma;
      }
    } else {
      // QR iteration: look for a small superdiagonal element.
      while (true) {
        int mm = l;
        for (; mm >= lend + 1; --mm) {
          if (wwr::fabs(E(mm - 1)) <= eps2 * wwr::fabs(D(mm) * D(mm - 1))) {
            break;
          }
        }
        if (mm < lend + 1) {
          mm = lend;
        }
        if (mm > lend) {
          E(mm - 1) = zero;
        }
        T p = D(l);
        if (mm == l) {
          // Eigenvalue found.
          D(l) = p;
          --l;
          if (l >= lend) {
            continue;
          }
          break;
        }
        // A remaining 2x2 block: ?lae2 gives its eigenvalues.
        if (mm == l - 1) {
          const T rte = wwr::sqrt(E(l - 1));
          T rt1, rt2;
          lae2_scalar(D(l), rte, D(l - 1), &rt1, &rt2);
          D(l) = rt1;
          D(l - 1) = rt2;
          E(l - 1) = zero;
          l -= 2;
          if (l >= lend) {
            continue;
          }
          break;
        }
        if (jtot == nmaxit) {
          break;
        }
        ++jtot;

        // Form the shift.
        T rte = wwr::sqrt(E(l - 1));
        T sigma = (D(l - 1) - p) / (two * rte);
        T r = lapy2_scalar(sigma, one);
        sigma = p - (rte / (sigma + wwr::copysign(r, sigma)));

        T c = one;
        T s = zero;
        T gamma = D(mm) - sigma;
        p = gamma * gamma;

        // Inner loop.
        const int lm1 = l - 1;
        for (int i = mm; i <= lm1; ++i) {
          const T bb = E(i);
          r = p + bb;
          if (i != mm) {
            E(i - 1) = s * r;
          }
          const T oldc = c;
          c = p / r;
          s = bb / r;
          const T oldgam = gamma;
          const T alpha = D(i + 1);
          gamma = c * (alpha - sigma) - s * oldgam;
          D(i) = oldgam + (alpha - gamma);
          if (c != zero) {
            p = (gamma * gamma) / c;
          } else {
            p = oldc * bb;
          }
        }
        E(lm1) = s * p;
        D(l) = sigma + gamma;
      }
    }

    // Undo scaling if necessary.
    if (iscale == 1) {
      lascl_g_serial(ssfmax, anorm, lendsv - lsv + 1, &D(lsv));
    } else if (iscale == 2) {
      lascl_g_serial(ssfmin, anorm, lendsv - lsv + 1, &D(lsv));
    }

    // No convergence after n*kMaxIt sweeps in total: count the unsplit e.
    if (jtot >= nmaxit) {
      for (int i = 1; i <= n - 1; ++i) {
        if (E(i) != zero) {
          ++info;
        }
      }
      return info;
    }
  }

  // Sort the eigenvalues in increasing order.
  lasrt_serial<T, SortDir::I>(d, n);
  return 0;
}

} // namespace calaman::device
