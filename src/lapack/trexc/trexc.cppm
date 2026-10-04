/**
 * @file trexc.cppm
 * @brief The calaman.trexc module -- reorder a real Schur factorization by
 *        moving a selected diagonal block to another position, LAPACK's ?trexc
 *
 * Moves the diagonal block of the n-by-n upper quasi-triangular matrix @p t (a
 * real Schur form) whose first row is @p ifst to the position whose first row is
 * @p ilst, by a sequence of adjacent-block swaps -- and, when @p wantq,
 * accumulates the orthogonal similarity into @p q. @p ifst and @p ilst are
 * 1-based and in/out: on entry they may point anywhere inside their block, and
 * each is first snapped to the first row of its block; on exit each holds where
 * its block ended up (as LAPACK's ?trexc leaves them). Templated over `float`
 * and `double`.
 *
 * It is a HOST COMPOSITION over calaman.laexc, the sibling of calaman.gehd2: the
 * block-reordering logic is sequential and index-dependent, so a host loop walks
 * the moving block one adjacent swap at a time -- each swap is one
 * calaman::laexc launch -- bubbling it up or down until it reaches @p ilst. The
 * loop reads back the one subdiagonal entry that tells a 1x1 block from a 2x2
 * (T(i+1,i) == 0 ?) and the device @p info laexc writes, so it owns no kernel of
 * its own.
 *
 * REAL ONLY: ?trexc swaps 2x2 blocks, which a complex Schur form (already
 * triangular) has none of, so LAPACK ships no complex ?trexc; the surface is
 * float / double, the scope the reference has.
 *
 * Allocation-free shipped surface (CLAUDE.md): @p t and @p q are caller-owned
 * device matrices, and @p dinfo is a caller-owned device int laexc writes each
 * swap (scratch, read back here). @p ifst / @p ilst / @p info are host ints --
 * the move is driven entirely from the host, like the gebal/gehd2 pipeline. A
 * rejected swap (laexc could not separate two too-close blocks) stops the move,
 * sets @p info = 1 and leaves @p ilst at the block's reached position.
 *
 * Usage:
 *   import calaman.trexc;     // also re-exports calaman::Status
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_t: N x N Schur form; d_q: N x N accumulator; d_info: device int
 *   int ifst = 5, ilst = 1, info = 0;
 *   calaman::trexc<double>(stream, true, n, d_t, ldt, d_q, ldq, &ifst, &ilst,
 *                          &info, d_info);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.trexc;

import wwr.runtime_api;     // wwrStream_t, wwrMemcpyAsync, wwrStreamSynchronize, wwrErrorInvalidValue
import wwr.wrappers.common; // real_fp -- the T constraint on the exported template
import calaman.laexc;       // calaman::laexc -- the adjacent-block swap each step runs
import std;                 // std::size_t

// export import, not a plain import: trexc RETURNS calaman::Status, so a consumer
// of `import calaman.trexc;` must see Status's member functions, not just its
// name -- the same re-export laexc / gehd2 do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// trexc_detail, not an anonymous namespace: these helpers are named by the
// exported trexc template's body, which is instantiated in every importer's TU.
// A named (unexported) namespace gives them module linkage, reachable by the
// instantiation yet absent from the public surface (as pstrf_detail does). The
// name is MODULE-SPECIFIC so a consumer importing trexc alongside another module
// with identically-shaped scalar helpers sees no one name on two modules.
namespace trexc_detail {

/// @brief Read one device T entry T(i,j) (1-based, column-major) to the host,
///        blocking on @p stream; writes it to @p out
template<typename T>
Status read_entry(wwr::wwrStream_t stream, const T *t, int ldt, int i, int j, T *out) {
  const T *p = t + (static_cast<std::size_t>(j - 1) * static_cast<std::size_t>(ldt) +
                    static_cast<std::size_t>(i - 1));
  CLM_TRY(wwr::wwrMemcpyAsync(out, p, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  return wwr::wwrSuccess;
}

/// @brief Read the device int @p dinfo to the host, blocking on @p stream
inline Status read_info(wwr::wwrStream_t stream, const int *dinfo, int *out) {
  CLM_TRY(wwr::wwrMemcpyAsync(out, dinfo, sizeof(int), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  return wwr::wwrSuccess;
}

/// @brief Is the subdiagonal entry T(i+1,i) nonzero? (i.e. the block at i is 2x2)
template<typename T>
Status subdiag_nonzero(wwr::wwrStream_t stream, const T *t, int ldt, int i, bool *out) {
  T v{};
  CLM_TRY(read_entry<T>(stream, t, ldt, i + 1, i, &v));
  *out = v != T{0};
  return wwr::wwrSuccess;
}

} // namespace trexc_detail

/// @brief Reorder a real Schur factorization by moving a diagonal block (?trexc)
///        on @p stream
///
/// Moves the block whose first row is @p ifst to the position @p ilst by adjacent
/// swaps (one calaman::laexc each), accumulating the transform into @p q when
/// @p wantq. @p ifst and @p ilst (1-based, in/out) are first snapped to their
/// block's first row and on exit hold the reached positions. @p info is 0 on a
/// completed move, 1 when a swap was rejected (two blocks too close to separate),
/// in which case the move stops and @p ilst marks how far the block got. A
/// quick return for @p n <= 1 writes only @p info = 0. @p t and @p q are
/// column-major device matrices; @p dinfo is a device int scratch laexc writes.
///
/// @tparam T Element type (float, double)
/// @param stream Stream the swaps are enqueued on; every device pointer lives on it
/// @param wantq Accumulate the orthogonal transform into @p q when true
/// @param n Order of @p t (and @p q when @p wantq), >= 0
/// @param t Device n-by-n Schur form, leading dimension @p ldt; reordered in place
/// @param ldt Leading dimension of @p t
/// @param q Device n-by-n accumulator, leading dimension @p ldq; touched only when @p wantq
/// @param ldq Leading dimension of @p q
/// @param ifst Host 1-based first row of the block to move; snapped and updated in place
/// @param ilst Host 1-based target first row; snapped and updated to the reached position
/// @param info Host int; 0 on success, 1 if a swap was rejected
/// @param dinfo Device int scratch; each laexc swap writes its 0/1 outcome here
/// @return Success, or the runtime error the first failing laexc launch reported
export template<wwr::real_fp T>
Status trexc(const wwr::wwrStream_t stream, const bool wantq, const int n, T *const t,
             const int ldt, T *const q, const int ldq, int *const ifst, int *const ilst,
             int *const info, int *const dinfo) {
  using namespace trexc_detail;
  *info = 0;

  // Argument validation, in LAPACK's ?trexc INFO order (-2, -4, -6, -7, -8).
  // IFST / ILST are only range-checked when n > 0, as the reference documents.
  const int one = 1;
  if (n < 0 || ldt < (n > one ? n : one) || ldq < one || (wantq && ldq < (n > one ? n : one)) ||
      (n > 0 && (*ifst < 1 || *ifst > n)) || (n > 0 && (*ilst < 1 || *ilst > n))) {
    return wwr::wwrErrorInvalidValue;
  }

  // Quick return. A 0/1-order matrix has nothing to reorder.
  if (n <= 1) {
    return wwr::wwrSuccess;
  }

  int ifst_ = *ifst;
  int ilst_ = *ilst;

  // Snap IFST to the first row of its block, and find whether it is 1x1 or 2x2.
  if (ifst_ > 1) {
    bool nz = false;
    CLM_TRY(subdiag_nonzero<T>(stream, t, ldt, ifst_ - 1, &nz)); // T(ifst, ifst-1) != 0
    if (nz) {
      --ifst_;
    }
  }
  int nbf = 1;
  if (ifst_ < n) {
    bool nz = false;
    CLM_TRY(subdiag_nonzero<T>(stream, t, ldt, ifst_, &nz)); // T(ifst+1, ifst) != 0
    if (nz) {
      nbf = 2;
    }
  }

  // Snap ILST to the first row of its block, and find whether it is 1x1 or 2x2.
  if (ilst_ > 1) {
    bool nz = false;
    CLM_TRY(subdiag_nonzero<T>(stream, t, ldt, ilst_ - 1, &nz)); // T(ilst, ilst-1) != 0
    if (nz) {
      --ilst_;
    }
  }
  int nbl = 1;
  if (ilst_ < n) {
    bool nz = false;
    CLM_TRY(subdiag_nonzero<T>(stream, t, ldt, ilst_, &nz)); // T(ilst+1, ilst) != 0
    if (nz) {
      nbl = 2;
    }
  }

  if (ifst_ == ilst_) {
    *ifst = ifst_;
    *ilst = ilst_;
    return wwr::wwrSuccess;
  }

  // One adjacent swap: laexc(j1, na, nb), then read its device info back. A
  // rejected swap (hinfo == 1) propagates to the caller through *info.
  auto swap = [&](int j1, int na, int nb, int *hinfo) -> Status {
    CLM_TRY(laexc<T>(stream, wantq, n, t, ldt, q, ldq, j1, na, nb, dinfo));
    CLM_TRY(read_info(stream, dinfo, hinfo));
    return wwr::wwrSuccess;
  };

  // The moving block's first-row position; on normal completion ilst_ := here.
  int here = ifst_;

  if (ifst_ < ilst_) {
    // Move the block DOWN. First correct ILST for the block-size change at the
    // destination: a 2x2 moving into a 1x1 target shifts the target up one, and
    // a 1x1 into a 2x2 target shifts it down one.
    if (nbf == 2 && nbl == 1) {
      --ilst_;
    }
    if (nbf == 1 && nbl == 2) {
      ++ilst_;
    }

    do {
      if (nbf == 1 || nbf == 2) {
        // Current block is 1x1 or 2x2; swap it with the next block below.
        int nbnext = 1;
        if (here + nbf + 1 <= n) {
          bool nz = false;
          CLM_TRY(subdiag_nonzero<T>(stream, t, ldt, here + nbf, &nz));
          if (nz) {
            nbnext = 2;
          }
        }
        int hinfo = 0;
        CLM_TRY(swap(here, nbf, nbnext, &hinfo));
        if (hinfo != 0) {
          *info = 1;
          break;
        }
        here += nbnext;
        // A 2x2 block may have split into two 1x1 blocks: switch to that path.
        if (nbf == 2) {
          bool nz = false;
          CLM_TRY(subdiag_nonzero<T>(stream, t, ldt, here, &nz));
          if (!nz) {
            nbf = 3;
          }
        }
      } else {
        // Two 1x1 blocks (from a split 2x2); swap each individually.
        int nbnext = 1;
        if (here + 3 <= n) {
          bool nz = false;
          CLM_TRY(subdiag_nonzero<T>(stream, t, ldt, here + 2, &nz));
          if (nz) {
            nbnext = 2;
          }
        }
        int hinfo = 0;
        CLM_TRY(swap(here + 1, 1, nbnext, &hinfo));
        if (hinfo != 0) {
          *info = 1;
          break;
        }
        if (nbnext == 1) {
          // Two 1x1 blocks: no rejection possible.
          CLM_TRY(swap(here, 1, nbnext, &hinfo));
          ++here;
        } else {
          // Recompute NBNEXT in case the 2x2 block split.
          bool nz = false;
          CLM_TRY(subdiag_nonzero<T>(stream, t, ldt, here + 1, &nz));
          if (!nz) {
            nbnext = 1;
          }
          if (nbnext == 2) {
            // The 2x2 block did not split.
            CLM_TRY(swap(here, 1, nbnext, &hinfo));
            if (hinfo != 0) {
              *info = 1;
              break;
            }
            here += 2;
          } else {
            // The 2x2 block split into two 1x1 blocks.
            CLM_TRY(swap(here, 1, 1, &hinfo));
            CLM_TRY(swap(here + 1, 1, 1, &hinfo));
            here += 2;
          }
        }
      }
    } while (here < ilst_);
  } else {
    do {
      if (nbf == 1 || nbf == 2) {
        // Current block is 1x1 or 2x2; swap it with the next block above.
        int nbnext = 1;
        if (here >= 3) {
          bool nz = false;
          CLM_TRY(subdiag_nonzero<T>(stream, t, ldt, here - 2, &nz));
          if (nz) {
            nbnext = 2;
          }
        }
        int hinfo = 0;
        CLM_TRY(swap(here - nbnext, nbnext, nbf, &hinfo));
        if (hinfo != 0) {
          *info = 1;
          break;
        }
        here -= nbnext;
        if (nbf == 2) {
          bool nz = false;
          CLM_TRY(subdiag_nonzero<T>(stream, t, ldt, here, &nz));
          if (!nz) {
            nbf = 3;
          }
        }
      } else {
        // Two 1x1 blocks (from a split 2x2); swap each individually.
        int nbnext = 1;
        if (here >= 3) {
          bool nz = false;
          CLM_TRY(subdiag_nonzero<T>(stream, t, ldt, here - 2, &nz));
          if (nz) {
            nbnext = 2;
          }
        }
        int hinfo = 0;
        CLM_TRY(swap(here - nbnext, nbnext, 1, &hinfo));
        if (hinfo != 0) {
          *info = 1;
          break;
        }
        if (nbnext == 1) {
          CLM_TRY(swap(here, nbnext, 1, &hinfo));
          --here;
        } else {
          // Recompute NBNEXT in case the 2x2 block split.
          bool nz = false;
          CLM_TRY(subdiag_nonzero<T>(stream, t, ldt, here - 1, &nz));
          if (!nz) {
            nbnext = 1;
          }
          if (nbnext == 2) {
            // The 2x2 block did not split.
            CLM_TRY(swap(here - 1, 2, 1, &hinfo));
            if (hinfo != 0) {
              *info = 1;
              break;
            }
            here -= 2;
          } else {
            CLM_TRY(swap(here, 1, 1, &hinfo));
            CLM_TRY(swap(here - 1, 1, 1, &hinfo));
            here -= 2;
          }
        }
      }
    } while (here > ilst_);
  }

  // Both a completed move and a rejected one leave ILST at HERE, the block's
  // reached first-row position (the reference's `ILST = HERE` on either exit).
  ilst_ = here;
  *ifst = ifst_;
  *ilst = ilst_;
  return wwr::wwrSuccess;
}

} // namespace calaman
