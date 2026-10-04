/**
 * @file interface.cppm
 * @brief Primary interface for calaman.gebak -- backward transformation of the
 *        eigenvectors of a balanced matrix, LAPACK's ?gebak
 *
 * Forms the eigenvectors of a general matrix A from the eigenvectors of the
 * balanced matrix B = D^-1 P^T A P D that calaman.gebal produced: undoes the
 * diagonal scaling D and the symmetric permutation P, in place, on the n-by-m
 * column-major matrix of right or left eigenvectors V. Templated over all four
 * element types (float, double, and the two complex types), constrained by
 * wwr::usual_fp. @p scale and @p ilo / @p ihi are exactly gebal's outputs, fed
 * through unchanged -- the dual encoding of @p scale (diagonal of D inside
 * [ilo, ihi], interchange index outside) is a fact of the ?gebak contract.
 *
 * Like gebal this is NOT a BLAS composition: both stages are hand-written kernels
 * (gebak.cu), reached through the launchers in gebak_bridge.h (included in the
 * GMF). But unlike gebal the DRIVER here reads NOTHING back from the device -- the
 * permutation is data- and order-dependent yet is replayed inside one cooperating
 * block on the device, so the driver is thin control flow, not a host loop over
 * device values.
 *
 * ASYNCHRONOUS. gebak only enqueues kernels on @p stream and never synchronizes;
 * on return the work may still be in flight. (This is the mirror image of gebal,
 * which must synchronize because it branches on device data.)
 *
 * COMPLEX is a first-class path for the same reason as gebal: the back-transform
 * only multiplies by the real scales D and moves rows, so there is no complex tau
 * or conjugation to differ. @p scale is therefore real even for complex T -- its
 * type is wwr::ComplexToRealType<T>, matching ?gebak's real SCALE argument.
 *
 * Differences from the reference ?gebak: CHARACTER JOB / SIDE become the typed
 * GebakJob / GebakSide enums, the s/d/c/z variants become one template over T, and
 * INFO becomes a returned calaman::Status (docs/architecture.md §4). The numerics
 * are identical -- there is no balancing-convention divergence here, only in gebal.
 *
 * Usage:
 *   import calaman.gebak;
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_V: n x m device matrix (ldv); d_scale, ilo, ihi: gebal's outputs
 *   calaman::gebak(stream, calaman::GebakJob::Both, calaman::GebakSide::Right, n,
 *                  ilo, ihi, d_scale, m, d_V, ldv);
 *   wwr::wwrStreamSynchronize(stream);   // gebak does NOT synchronize
 */

module;

#include "gebak_bridge.h"
// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the export import
// below supplies.
#include "error_handling/error_macros.h"

export module calaman.gebak;

import std;
import wwr.runtime_api;      // wwrStream_t, wwrError_t, wwrSuccess, wwrGetLastError
import wwr.complex;          // wwrFloatComplex, wwrDoubleComplex (extern template list)
import wwr.wrappers.common;  // usual_fp, ComplexToRealType
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

/// @brief Which halves of the back-transformation to run -- LAPACK's JOB argument
///
/// None    ('N'): do nothing, return immediately.
/// Permute ('P'): undo the permutation P only.
/// Scale   ('S'): undo the diagonal scaling D only.
/// Both    ('B'): undo both. Must match the JOB passed to gebal.
export enum class GebakJob { None, Permute, Scale, Both };

/// @brief Whether V holds right or left eigenvectors -- LAPACK's SIDE argument
///
/// Right ('R'): scale row i by scale(i).   Left ('L'): scale row i by 1/scale(i).
export enum class GebakSide { Right, Left };

// Launch-error checks early-return the first failure via the shared CLM_TRY
// (error_handling/error_macros.h), included in the GMF above. It is a plain
// `#pragma once` header macro, intentionally not #undef'd -- there is no local
// macro here to leak.

/// @brief Back-transform eigenvectors of a balanced matrix (?gebak)
///
/// Undoes, on the eigenvectors in @p d_V, the similarity gebal applied: first the
/// backward scaling (rows ilo..ihi multiplied by scale(i) for right vectors,
/// 1/scale(i) for left), then the backward permutation (rows outside [ilo, ihi]
/// interchanged by the indices stored in @p d_scale), exactly in ?gebak's order.
/// A single-index window (ilo == ihi) carries no scaling, matching LAPACK.
///
/// ASYNCHRONOUS: enqueues its kernels on @p stream and returns without
/// synchronizing. The caller must synchronize before reading @p d_V on the host.
///
/// @tparam T Element type; one of the instantiated types (the four usual_fp types)
/// @param stream  Stream the kernels launch on; V and scale live on its device
/// @param job     Which halves of the back-transformation to run (match gebal's)
/// @param side    Whether V holds right or left eigenvectors
/// @param n       Number of rows of V (the balanced matrix dimension)
/// @param ilo     1-based first index of gebal's balanced middle block
/// @param ihi     1-based last index of it; 1 <= ilo <= ihi <= n for n > 0,
///                and ilo = 1, ihi = 0 for n = 0
/// @param d_scale Device array length n, real even for complex T: gebal's output,
///                diagonal of D inside [ilo, ihi] and the 1-based interchange index
///                outside it. Read only where @p job needs it.
/// @param m       Number of columns of V (number of eigenvectors)
/// @param d_V     Device n-by-m matrix, column-major, overwritten in place
/// @param ldv     Leading dimension of d_V (>= max(1, n))
/// @return A success Status, or the first runtime error encountered
export template<wwr::usual_fp T>
Status gebak(wwr::wwrStream_t stream, const GebakJob job, const GebakSide side, const int n,
             const int ilo, const int ihi, const wwr::ComplexToRealType<T> *d_scale, const int m,
             T *d_V, const int ldv) {
  using R = wwr::ComplexToRealType<T>;

  const bool do_permute = (job == GebakJob::Permute) || (job == GebakJob::Both);
  const bool do_scale = (job == GebakJob::Scale) || (job == GebakJob::Both);
  const bool leftv = (side == GebakSide::Left);

  // Argument tests mirroring ?gebak: the index bounds are LAPACK's exactly
  // (1 <= ilo <= max(1,n); min(ilo,n) <= ihi <= n), collapsed to one Status.
  const int max1n = (n > 1) ? n : 1;
  const int ihi_lo = (ilo < n) ? ilo : n;
  if (n < 0 || m < 0 || ldv < max1n || ilo < 1 || ilo > max1n || ihi < ihi_lo || ihi > n) {
    return wwr::wwrErrorInvalidValue;
  }

  // Quick returns, after validation, in ?gebak's order.
  if (n == 0 || m == 0 || job == GebakJob::None) {
    return wwr::wwrSuccess;
  }
  if (d_V == nullptr || d_scale == nullptr) {
    return wwr::wwrErrorInvalidValue;
  }

  // Backward scaling. A single-index window has nothing to scale (LAPACK's
  // `IF (ilo == ihi) GO TO 30`); the window is the 0-based [ilo-1, ihi-1].
  if (do_scale && ilo != ihi) {
    device::gebak_scale_rows<T, R>(stream, d_V, ldv, m, ilo - 1, ihi - 1, d_scale, leftv);
    CLM_TRY(wwr::wwrGetLastError());
  }

  // Backward permutation, replayed on the device in one block (ilo/ihi 1-based).
  if (do_permute) {
    device::gebak_permute<T, R>(stream, d_V, ldv, m, n, ilo, ihi, d_scale);
    CLM_TRY(wwr::wwrGetLastError());
  }

  return wwr::wwrSuccess;
}

// Paired with instantiations.cpp: gebak is instantiated once inside this library
// (its body names the .cu-side launchers declared only in the GMF), so an importer
// never re-instantiates it.
extern template Status gebak<float>(wwr::wwrStream_t, GebakJob, GebakSide, int, int, int,
                                    const float *, int, float *, int);
extern template Status gebak<double>(wwr::wwrStream_t, GebakJob, GebakSide, int, int, int,
                                     const double *, int, double *, int);
extern template Status gebak<wwr::wwrFloatComplex>(wwr::wwrStream_t, GebakJob, GebakSide, int, int,
                                                   int, const float *, int, wwr::wwrFloatComplex *,
                                                   int);
extern template Status gebak<wwr::wwrDoubleComplex>(wwr::wwrStream_t, GebakJob, GebakSide, int, int,
                                                    int, const double *, int,
                                                    wwr::wwrDoubleComplex *, int);

} // namespace calaman
