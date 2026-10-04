/**
 * @file laqr0.cppm
 * @brief The calaman.laqr0 module -- the small-bulge multishift QR driver with
 *        aggressive early deflation for the Schur form of a Hessenberg matrix,
 *        LAPACK's ?laqr0
 *
 * Computes the real Schur form of the active window @p ilo..@p ihi of an upper
 * Hessenberg matrix @p h by the Braman-Byers-Mathias multishift QR iteration:
 * each sweep deflates the trailing window with calaman.laqr3 (the recursive AED,
 * whose window Schur form is the multishift calaman.laqr4 when the window exceeds
 * the ILAENV(12) crossover NMIN = 75, else calaman.lahqr), takes its undeflated
 * eigenvalues (or a trailing-submatrix calaman.laqr4) as shifts, and chases them
 * through the block with calaman.laqr5. Tiny windows (@p n <= NTINY = 15) fall
 * straight to calaman.lahqr, as the reference's hard subdiagonal-scratch limit
 * requires. REAL ONLY (float / double).
 *
 * ?laqr0 IS ?laqr4 EXCEPT for two calls: the AED subroutine is the recursive,
 * more reliable ?laqr3 (where ?laqr4 uses the small-window ?laqr2), and the
 * trailing-submatrix shift recomputation is ?laqr4 (where ?laqr4 uses ?lahqr).
 * Everything else -- the tiny-window base case, window sizing, the nibble
 * heuristic, exceptional shifts, the shift sort / shuffle, the sweep -- is the
 * same loop. This port therefore reuses calaman.laqr4's host driver verbatim and
 * swaps only those two kernels.
 *
 * A HOST COMPOSITION, not a kernel: the driver loop (active-block location,
 * deflation-window sizing, shift selection / sort / shuffle) runs on the host and
 * stages the handful of scalars it inspects through device memcpies, exactly as
 * calaman.gehrd drives its panel kernels. The heavy steps are the laqr3 / laqr4 /
 * laqr5 kernels; this module adds no device code. Because those kernels take
 * their scratch from the caller-owned @p work here (not aliased into the strictly
 * lower corner of @p h the reference uses), @p h's output below the first
 * subdiagonal stays clean -- the only place this port's @p h differs from the
 * reference's, which leaves AED leftovers there.
 *
 * Allocation-free shipped surface (CLAUDE.md): @p h, @p wr, @p wi, @p z, @p work
 * and @p info are caller-provided device pointers. @p lwork governs NWMAX / NSMAX
 * exactly as the reference does (pass the same value to both to match shift
 * schedules); @p work must additionally hold this port's scratch -- see below. It
 * carries, over calaman.laqr4's layout, one extra region: the recursive-laqr4
 * scratch calaman.laqr3 and the shift recomputation both consume. KACC22 is
 * pinned to 0 (calaman.laqr5's only mode); the reference's KACC22 >= 1 for
 * NS >= 14 differs only in gemm-accumulation order, within tolerance.
 *
 * Usage:
 *   import calaman.laqr0;
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   // d_h: n x n Hessenberg; d_z: n x n accumulator; d_wr/d_wi: length n
 *   calaman::laqr0<double>(stream, true, true, n, 1, n, d_h, ldh, d_wr, d_wi,
 *                          1, n, d_z, ldz, d_work, lwork, d_info);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.laqr0;

import wwr.runtime_api;   // wwrStream_t, wwrMemcpyAsync, wwrStreamSynchronize
import wwr.wrappers.common; // real_fp
import calaman.lahqr;     // calaman::lahqr -- the tiny-window base case
import calaman.laqr3;     // calaman::laqr3 -- the recursive aggressive early deflation
import calaman.laqr4;     // calaman::laqr4 -- the trailing-submatrix shift source
import calaman.laqr5;     // calaman::laqr5 -- the multishift bulge-chasing sweep
import std;

// export import, not a plain import: laqr0 RETURNS calaman::Status, so a consumer
// of `import calaman.laqr0;` must see Status's member functions, not just its
// name -- the same re-export laqr3 / laqr4 / laqr5 / lahqr do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// The driver's internal helpers sit at module (not anonymous-namespace) scope so
// the exported laqr0<T> template can name them when an importer instantiates its
// body -- an internal-linkage entity would be unreachable there. They are not
// exported, so they carry module linkage, like calaman.gehrd's kGehrdBlockSize.
//
// They live in a laqr0_detail NESTED namespace rather than bare calaman:: on
// purpose: laqr0 imports calaman.laqr4, which declares the SAME-named module
// helpers (kNtiny, iparmq_ns, host_dlanv2, ...) as its own module-linkage inline
// entities. Two inline/template entities of one qualified name attached to two
// modules is ill-formed ("cannot be attached to other modules"), so the nested
// namespace gives laqr0's copies a distinct qualified name -- the same clash the
// calaman.laqr3 interface sidesteps by making its kNmin function-local.
namespace laqr0_detail {

// ?laqr0's exceptional-shift / window / crossover constants, the reference's
// hard-coded PARAMETERs plus the ILAENV(IPARMQ) values the netlib build returns.
template<typename T>
inline constexpr T kWilk1 = T{0.75};
template<typename T>
inline constexpr T kWilk2 = T{-0.4375};
inline constexpr int kNtiny = 15; // hard subdiagonal-scratch limit: below -> lahqr
inline constexpr int kKexnw = 5;  // vary the window after this many no-deflation steps
inline constexpr int kKexsh = 6;  // try exceptional shifts every this many steps
inline constexpr int kNmin = 75;  // ILAENV(12): the lahqr crossover (nibble guard)
inline constexpr int kNibble = 14; // ILAENV(14): skip-the-sweep nibble threshold

// IPARMQ's ISHFTS (ISPEC=15) recommended shift count for a window of order nh --
// the stepwise schedule the reference consults; NWR (INWIN) and NSR share it.
int iparmq_ns(const int nh) {
  int ns = 2;
  if (nh >= 30) {
    ns = 4;
  }
  if (nh >= 60) {
    ns = 10;
  }
  if (nh >= 150) {
    ns = std::max(10, nh / static_cast<int>(std::lround(std::log(static_cast<double>(nh)) /
                                                        std::log(2.0))));
  }
  if (nh >= 590) {
    ns = 64;
  }
  if (nh >= 3000) {
    ns = 128;
  }
  if (nh >= 6000) {
    ns = 256;
  }
  return std::max(2, ns - ns % 2);
}

// The length of the recursive-laqr4 scratch region: laqr4's own @p work, sized
// for a subproblem of order @p sub at the shared @p lwork. laqr0 needs it twice
// -- as the @p work calaman.laqr3 forwards to its window laqr4, and as the @p
// work of the trailing-submatrix shift recomputation -- both bounded by the
// NWMAX window, so one region of this length (at sub = NWMAX) serves both. The
// formula is calaman.laqr4's own carve: 2 V/T windows + the NS-by-NS shift copy
// + a 1x1 Z dummy + 3 device counters.
std::size_t laqr4_workbuf_len(const int sub, const int lwork) {
  const int nwmax = std::min((sub - 1) / 3, lwork / 2);
  int nsmax = std::min((sub - 3) / 6, 2 * lwork / 3);
  nsmax -= nsmax % 2;
  const int ldv = std::max(1, nwmax);
  const int ldsc = std::max(1, nsmax);
  return 2 * static_cast<std::size_t>(ldv) * std::max(1, nwmax) +
         static_cast<std::size_t>(ldsc) * std::max(1, nsmax) + 4;
}

// Host port of LAPACK ?lanv2, eigenvalues only: standardise the real 2x2
// [a b; c d] and return its two eigenvalues (rt1r,rt1i) and (rt2r,rt2i). The
// driver forms exceptional shifts and the rare-failure fallback from these on
// the host, so -- unlike the device dlanv2 inlined in laqr2.cu -- it uses std::
// math and discards the rotation. V. Sima's cancellation-reducing revision, the
// arithmetic calaman.lanv2 and the reference share.
template<typename T>
void host_dlanv2(T a, T b, T c, T d, T &rt1r, T &rt1i, T &rt2r, T &rt2i) {
  const T zero{0}, half{0.5}, one{1}, two{2}, multpl{4};
  const T eps = std::numeric_limits<T>::epsilon();
  const T safmin = std::numeric_limits<T>::min();
  const T base = T{2};
  const T safmn2 = std::pow(base, static_cast<T>(static_cast<int>(
                                      std::log(safmin / eps) / std::log(base) / two)));
  const T safmx2 = one / safmn2;
  T cs, sn;
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
  } else if ((a - d) == zero && std::copysign(one, b) != std::copysign(one, c)) {
    cs = one;
    sn = zero;
  } else {
    T temp = a - d;
    T p = half * temp;
    const T bcmax = std::max(std::abs(b), std::abs(c));
    const T bcmis =
        std::min(std::abs(b), std::abs(c)) * std::copysign(one, b) * std::copysign(one, c);
    T scale = std::max(std::abs(p), bcmax);
    T z = (p / scale) * p + (bcmax / scale) * bcmis;
    if (z >= multpl * eps) {
      z = p + std::copysign(std::sqrt(scale) * std::sqrt(z), p);
      a = d + z;
      d = d - (bcmax / z) * bcmis;
      const T tau = std::hypot(c, z);
      cs = z / tau;
      sn = c / tau;
      b = b - c;
      c = zero;
    } else {
      T sigma = b + c;
      for (int count = 1;; ++count) {
        scale = std::max(std::abs(temp), std::abs(sigma));
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
      T tau = std::hypot(sigma, temp);
      cs = std::sqrt(half * (one + std::abs(sigma) / tau));
      sn = -(p / (tau * cs)) * std::copysign(one, sigma);
      const T aa = a * cs + b * sn;
      const T bb = -a * sn + b * cs;
      const T cc = c * cs + d * sn;
      const T dd = -c * sn + d * cs;
      a = aa * cs + cc * sn;
      b = bb * cs + dd * sn;
      c = -aa * sn + cc * cs;
      d = -bb * sn + dd * cs;
      temp = half * (a + d);
      a = temp;
      d = temp;
      if (c != zero) {
        if (b != zero) {
          if (std::copysign(one, b) == std::copysign(one, c)) {
            const T sab = std::sqrt(std::abs(b));
            const T sac = std::sqrt(std::abs(c));
            p = std::copysign(sab * sac, c);
            a = temp + p;
            d = temp - p;
            b = b - c;
            c = zero;
          }
        } else {
          b = -c;
          c = zero;
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
    rt1i = std::sqrt(std::abs(b)) * std::sqrt(std::abs(c));
    rt2i = -rt1i;
  }
}

} // namespace laqr0_detail

/// @brief Small-bulge multishift QR with aggressive early deflation (?laqr0)
///
/// Enqueues the multishift QR iteration on the active window @p ilo..@p ihi
/// (1-based) of the Hessenberg @p h and synchronizes internally (the host loop
/// inspects device scalars between kernels). On return the window of @p h is
/// quasi-triangular (the full Schur form when @p wantt, below the first
/// subdiagonal left clean), @p wr / @p wi hold the eigenvalues, the orthogonal
/// transform is accumulated into rows @p iloz..@p ihiz of @p z when @p wantz,
/// and @p info (device int) is 0 on success or the index below which convergence
/// failed. @p n <= NTINY = 15 is handled by calaman.lahqr directly. No argument
/// checking beyond an @p n <= 0 quick return (info := 0); the workspace-query
/// path (@p lwork = -1) is not supported by this allocation-free port.
///
/// @tparam T Element type (float, double)
/// @param stream Stream the launches are enqueued on; every pointer lives on its device
/// @param wantt Compute the full Schur form of @p h when true, else eigenvalues only
/// @param wantz Accumulate the orthogonal transform into @p z when true
/// @param n Order of @p h (and @p z when @p wantz)
/// @param ilo 1-based first index of the active window
/// @param ihi 1-based last index of the active window
/// @param h Device N-by-N upper Hessenberg matrix, leading dimension @p ldh; updated in place
/// @param ldh Leading dimension of @p h
/// @param wr Device length->= @p ihi real parts of the eigenvalues
/// @param wi Device length->= @p ihi imaginary parts of the eigenvalues
/// @param iloz 1-based first row of @p z the transform touches
/// @param ihiz 1-based last row of @p z the transform touches
/// @param z Device N-by-N accumulator, leading dimension @p ldz; touched only when @p wantz
/// @param ldz Leading dimension of @p z
/// @param work Device scratch; length >= 2*NWMAX*NWMAX + NSMAX*NSMAX + 1 +
///        laqr4_workbuf_len(NWMAX, @p lwork) + 4, with NWMAX = min((@p n -1)/3,
///        @p lwork /2) and NSMAX the even min((@p n -3)/6, 2*@p lwork /3) (holds
///        the laqr3 V/T windows, the laqr4 shift-submatrix copy, a 1x1 Z dummy,
///        the recursive-laqr4 scratch, and three device counters)
/// @param lwork Governs NWMAX / NSMAX as in the reference; pass the value the oracle gets
/// @param info Device int; 0 on success, or the index below which convergence failed
/// @return Success, or the runtime error a kernel launch or memcpy reported
export template<wwr::real_fp T>
Status laqr0(const wwr::wwrStream_t stream, const bool wantt, const bool wantz, const int n,
             const int ilo, const int ihi, T *const h, const int ldh, T *const wr, T *const wi,
             const int iloz, const int ihiz, T *const z, const int ldz, T *const work,
             const int lwork, int *const info) {
  using std::size_t;
  using namespace laqr0_detail; // kNtiny, kKexnw/sh, kNmin, kNibble, kWilk1/2,
                                // iparmq_ns, laqr4_workbuf_len, host_dlanv2
  const T zero{0};

  if (n <= 0) {
    const int z0 = 0;
    CLM_TRY(wwr::wwrMemcpyAsync(info, &z0, sizeof(int), wwr::wwrMemcpyHostToDevice, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    return wwr::wwrSuccess;
  }

  // Device-scalar staging helpers, the calaman.gehrd idiom: one async copy then a
  // synchronize, so the host sees a settled value. Small and sync-per-call --
  // fine here, where the loop inspects only O(1) entries of H and the shift arrays
  // between O(n^2) kernels. Hptr folds the column-major leading dimension out.
  auto Hptr = [=](int i, int j) -> T * {
    return h + static_cast<size_t>(i - 1) + static_cast<size_t>(j - 1) * ldh;
  };
  auto read_dev = [&](T *dst, const T *src, int count) -> Status {
    CLM_TRY(wwr::wwrMemcpyAsync(dst, src, static_cast<size_t>(count) * sizeof(T),
                               wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    return wwr::wwrSuccess;
  };
  auto write_dev = [&](T *dst, const T *src, int count) -> Status {
    CLM_TRY(wwr::wwrMemcpyAsync(dst, src, static_cast<size_t>(count) * sizeof(T),
                               wwr::wwrMemcpyHostToDevice, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    return wwr::wwrSuccess;
  };
  // getH is used in scalar context; fold the Status check into a throwing-free
  // path by propagating through the outer CLM_TRY sites that call read_dev.
  auto getH = [&](int i, int j, T &out) -> Status { return read_dev(&out, Hptr(i, j), 1); };

  // Carve the scratch. NWMAX / NSMAX mirror the reference so the window and shift
  // schedules match for the same lwork; work holds the laqr3 V/T windows, the
  // laqr4 shift-submatrix copy, a 1x1 Z dummy, the recursive-laqr4 scratch, then
  // three device ints. The recursive-laqr4 region (iwork) serves both the laqr3
  // AED's inner window laqr4 and the trailing-submatrix shift recomputation --
  // never live at the same time -- so one region at the NWMAX bound suffices.
  const int nwmax = std::min((n - 1) / 3, lwork / 2);
  int nsmax = std::min((n - 3) / 6, 2 * lwork / 3);
  nsmax -= nsmax % 2;
  const int ldv = std::max(1, nwmax);
  const int ldsc = std::max(1, nsmax);
  const size_t iworklen = laqr4_workbuf_len(nwmax, lwork);
  size_t off = 0;
  T *const vbuf = work + off;
  off += static_cast<size_t>(ldv) * std::max(1, nwmax);
  T *const tbuf = work + off;
  off += static_cast<size_t>(ldv) * std::max(1, nwmax);
  T *const scbuf = work + off;
  off += static_cast<size_t>(ldsc) * std::max(1, nsmax);
  T *const zdum = work + off;
  off += 1;
  T *const iwork = work + off;
  off += iworklen;
  int *const d_counts = reinterpret_cast<int *>(work + off);
  int *const d_ns = &d_counts[0];
  int *const d_nd = &d_counts[1];
  int *const d_info = &d_counts[2];

  // ==== Tiny matrices must use lahqr (hard subdiagonal-scratch limit). ====
  if (n <= kNtiny) {
    CLM_TRY(
        lahqr<T>(stream, wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, iloz, ihiz, z, ldz, d_info));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    int inf = 0;
    CLM_TRY(wwr::wwrMemcpyAsync(&inf, d_info, sizeof(int), wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    CLM_TRY(wwr::wwrMemcpyAsync(info, &inf, sizeof(int), wwr::wwrMemcpyHostToDevice, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    return wwr::wwrSuccess;
  }

  // ==== Recommended window (NWR) and shift (NSR) counts, clamped. ====
  const int nh0 = ihi - ilo + 1;
  const int ns_reco = iparmq_ns(nh0);
  int nwr = (nh0 <= 500) ? ns_reco : 3 * ns_reco / 2;
  nwr = std::max(2, nwr);
  nwr = std::min({ihi - ilo + 1, (n - 1) / 3, nwr});
  int nsr = std::min({ns_reco, (n - 3) / 6, ihi - ilo});
  nsr = std::max(2, nsr - nsr % 2);

  int nw = nwmax;
  int ndfl = 1; // iteration count, restarted at each deflation
  int ndec = 0; // exceptional-window shrink counter (reference leaves uninit; -1 below)
  const int itmax = std::max(30, 2 * kKexsh) * std::max(10, ihi - ilo + 1);
  int kbot = ihi;
  int hinfo = 0; // host INFO: 0, or KBOT if the iteration limit is hit

  // Host mirrors of the shift arrays over the active trailing range.
  std::vector<T> hwr(static_cast<size_t>(n)), hwi(static_cast<size_t>(n));

  for (int it = 1; it <= itmax; ++it) {
    if (kbot < ilo) {
      break;
    }

    // ==== Locate the active block: scan the subdiagonal up from KBOT. ====
    int ktop = ilo;
    for (int k = kbot; k >= ilo + 1; --k) {
      T sub{};
      CLM_TRY(getH(k, k - 1, sub));
      if (sub == zero) {
        ktop = k;
        break;
      }
    }

    // ==== Select the deflation window size (typical + exceptional). ====
    const int nhb = kbot - ktop + 1;
    const int nwupbd = std::min(nhb, nwmax);
    if (ndfl < kKexnw) {
      nw = std::min(nwupbd, nwr);
    } else {
      nw = std::min(nwupbd, 2 * nw);
    }
    if (nw < nwmax) {
      if (nw >= nhb - 1) {
        nw = nhb;
      } else {
        const int kwtop = kbot - nw + 1;
        T a{}, b{};
        CLM_TRY(getH(kwtop, kwtop - 1, a));
        CLM_TRY(getH(kwtop - 1, kwtop - 2, b));
        if (std::abs(a) > std::abs(b)) {
          nw = nw + 1;
        }
      }
    }
    if (ndfl < kKexnw) {
      ndec = -1;
    } else if (ndec >= 0 || nw >= nwupbd) {
      ndec = ndec + 1;
      if (nw - ndec < 2) {
        ndec = 0;
      }
      nw = nw - ndec;
    }

    // ==== Aggressive early deflation (calaman.laqr3, the recursive variant).
    //      Its window Schur form uses the recursive calaman.laqr4 when the window
    //      exceeds NMIN = 75; iwork / lwork are that laqr4's scratch. ====
    CLM_TRY(laqr3<T>(stream, wantt, wantz, n, ktop, kbot, nw, h, ldh, iloz, ihiz, z, ldz, d_ns,
                     d_nd, wr, wi, vbuf, ldv, n, tbuf, ldv, n, vbuf, ldv, iwork, lwork));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    int ls = 0, ld = 0;
    CLM_TRY(wwr::wwrMemcpyAsync(&ls, d_ns, sizeof(int), wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(wwr::wwrMemcpyAsync(&ld, d_nd, sizeof(int), wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));

    kbot = kbot - ld;
    int ks = kbot - ls + 1;

    // ==== Skip an expensive sweep when the nibble heuristic expects more
    //      deflations, or the active block is already small. ====
    if (ld == 0 ||
        (100 * ld <= nw * kNibble && kbot - ktop + 1 > std::min(kNmin, nwmax))) {
      int ns = std::min({nsmax, nsr, std::max(2, kbot - ktop)});
      ns -= ns % 2;

      if (ndfl % kKexsh == 0) {
        // ==== Exceptional shifts, formed on the host via ?lanv2. ====
        ks = kbot - ns + 1;
        for (int i = kbot; i >= std::max(ks + 1, ktop + 2); i -= 2) {
          T hii1{}, hi1i2{}, hii{};
          CLM_TRY(getH(i, i - 1, hii1));
          CLM_TRY(getH(i - 1, i - 2, hi1i2));
          CLM_TRY(getH(i, i, hii));
          const T ss = std::abs(hii1) + std::abs(hi1i2);
          const T aa = kWilk1<T> * ss + hii;
          const T bb = ss;
          const T cc = kWilk2<T> * ss;
          const T dd = aa;
          host_dlanv2(aa, bb, cc, dd, hwr[i - 2], hwi[i - 2], hwr[i - 1], hwi[i - 1]);
        }
        if (ks == ktop) {
          T hkk{};
          CLM_TRY(getH(ks + 1, ks + 1, hkk));
          hwr[ks] = hkk; // WR(KS+1)
          hwi[ks] = zero;
          hwr[ks - 1] = hwr[ks]; // WR(KS) = WR(KS+1)
          hwi[ks - 1] = hwi[ks];
        }
      } else {
        // ==== Too few shifts from AED? recompute from a trailing NS-by-NS
        //      submatrix with calaman.laqr4 (eigenvalues only) -- the recursive
        //      multishift QR, where calaman.laqr4's driver uses calaman.lahqr. ====
        if (kbot - ks + 1 <= ns / 2) {
          ks = kbot - ns + 1;
          // DLACPY('A', NS, NS, H(KS,KS), LDH, scbuf, LDSC): copy column by column.
          for (int j = 0; j < ns; ++j) {
            CLM_TRY(wwr::wwrMemcpyAsync(scbuf + static_cast<size_t>(j) * ldsc,
                                       Hptr(ks, ks + j), static_cast<size_t>(ns) * sizeof(T),
                                       wwr::wwrMemcpyDeviceToDevice, stream));
          }
          CLM_TRY(wwr::wwrStreamSynchronize(stream));
          CLM_TRY(laqr4<T>(stream, false, false, ns, 1, ns, scbuf, ldsc, wr + (ks - 1),
                           wi + (ks - 1), 1, 1, zdum, 1, iwork, lwork, d_info));
          CLM_TRY(wwr::wwrStreamSynchronize(stream));
          int inf = 0;
          CLM_TRY(wwr::wwrMemcpyAsync(&inf, d_info, sizeof(int), wwr::wwrMemcpyDeviceToHost,
                                     stream));
          CLM_TRY(wwr::wwrStreamSynchronize(stream));
          // Mirror the eigenvalues laqr4 wrote over [ks, kbot] into the host arrays.
          CLM_TRY(read_dev(&hwr[ks - 1], wr + (ks - 1), kbot - ks + 1));
          CLM_TRY(read_dev(&hwi[ks - 1], wi + (ks - 1), kbot - ks + 1));
          ks = ks + inf;
          if (ks >= kbot) {
            // Rare QR failure: use the trailing 2x2's eigenvalues.
            T aa{}, cc{}, bb{}, dd{};
            CLM_TRY(getH(kbot - 1, kbot - 1, aa));
            CLM_TRY(getH(kbot, kbot - 1, cc));
            CLM_TRY(getH(kbot - 1, kbot, bb));
            CLM_TRY(getH(kbot, kbot, dd));
            host_dlanv2(aa, bb, cc, dd, hwr[kbot - 2], hwi[kbot - 2], hwr[kbot - 1], hwi[kbot - 1]);
            ks = kbot - 1;
          }
        } else {
          // AED already supplied the shifts on the device; mirror them.
          CLM_TRY(read_dev(&hwr[ks - 1], wr + (ks - 1), kbot - ks + 1));
          CLM_TRY(read_dev(&hwi[ks - 1], wi + (ks - 1), kbot - ks + 1));
        }

        if (kbot - ks + 1 > ns) {
          // ==== Sort the shifts (bubble sort keeps conjugate pairs together). ====
          bool sorted = false;
          for (int k = kbot; k >= ks + 1 && !sorted; --k) {
            sorted = true;
            for (int i = ks; i <= k - 1; ++i) {
              if (std::abs(hwr[i - 1]) + std::abs(hwi[i - 1]) <
                  std::abs(hwr[i]) + std::abs(hwi[i])) {
                sorted = false;
                std::swap(hwr[i - 1], hwr[i]);
                std::swap(hwi[i - 1], hwi[i]);
              }
            }
          }
        }

        // ==== Shuffle into real pairs and conjugate pairs. ====
        for (int i = kbot; i >= ks + 2; i -= 2) {
          if (hwi[i - 1] != -hwi[i - 2]) {
            const T sr0 = hwr[i - 1];
            hwr[i - 1] = hwr[i - 2];
            hwr[i - 2] = hwr[i - 3];
            hwr[i - 3] = sr0;
            const T si0 = hwi[i - 1];
            hwi[i - 1] = hwi[i - 2];
            hwi[i - 2] = hwi[i - 3];
            hwi[i - 3] = si0;
          }
        }
      }

      // ==== Two real shifts? keep only the better one. ====
      if (kbot - ks + 1 == 2) {
        if (hwi[kbot - 1] == zero) {
          T hkk{};
          CLM_TRY(getH(kbot, kbot, hkk));
          if (std::abs(hwr[kbot - 1] - hkk) < std::abs(hwr[kbot - 2] - hkk)) {
            hwr[kbot - 2] = hwr[kbot - 1];
          } else {
            hwr[kbot - 1] = hwr[kbot - 2];
          }
        }
      }

      // ==== Use up to NS of the smallest shifts (even count). ====
      ns = std::min(ns, kbot - ks + 1);
      ns -= ns % 2;
      ks = kbot - ns + 1;

      // Push the chosen shifts back to the device, then chase them.
      CLM_TRY(write_dev(wr + (ks - 1), &hwr[ks - 1], ns));
      CLM_TRY(write_dev(wi + (ks - 1), &hwi[ks - 1], ns));
      CLM_TRY(laqr5<T>(stream, wantt, wantz, n, ktop, kbot, ns, wr + (ks - 1), wi + (ks - 1), h,
                       ldh, iloz, ihiz, z, ldz));
      CLM_TRY(wwr::wwrStreamSynchronize(stream));
    }

    // ==== Note progress (or the lack of it). ====
    if (ld > 0) {
      ndfl = 1;
    } else {
      ndfl = ndfl + 1;
    }

    if (it == itmax) {
      hinfo = kbot; // iteration limit exceeded
    }
  }

  CLM_TRY(wwr::wwrMemcpyAsync(info, &hinfo, sizeof(int), wwr::wwrMemcpyHostToDevice, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  return wwr::wwrSuccess;
}

} // namespace calaman
