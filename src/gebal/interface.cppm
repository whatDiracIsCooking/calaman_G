/**
 * @file interface.cppm
 * @brief Primary interface for calaman.gebal -- matrix balancing for the
 *        nonsymmetric eigenproblem, LAPACK's ?gebal
 *
 * Overwrites an n-by-n column-major matrix A with B = D^-1 P^T A P D, isolating
 * eigenvalues that already sit on the diagonal (a symmetric permutation P) and
 * equilibrating what is left (a powers-of-two diagonal D, Parlett-Reinsch) so a
 * subsequent Hessenberg reduction and QR iteration see far smaller eigenvalue
 * condition numbers. Templated over all four element types (float, double, and
 * the two complex types), constrained by wwr::usual_fp.
 *
 * Unlike the geqp3 call graph, gebal is NOT a BLAS composition: both stages are
 * hand-written kernels (gebal.cu). But the DRIVER is a host composition and
 * lives here -- it branches on device data every step (which row isolates next,
 * whether a sweep changed anything), so it reads the device back with
 * wwr.runtime_api between launches and is host-driven by construction. The
 * kernels are reached through the launchers in gebal_bridge.h (included in the
 * GMF), the same module/.cu split calaman.lacpy uses; a wwrGetLastError() after
 * each launch keeps the reference driver's error discipline.
 *
 * COMPLEX, unlike larfg/laqp2, is a genuine first-class path here, not a
 * deferred extension: balancing only measures magnitudes (CABS1, |Re| + |Im|,
 * matching CGEBAL/ZGEBAL) and applies real power-of-two scales, so there is no
 * complex reflector or conjugated row to differ materially. @p d_scale is
 * therefore real even for complex T -- its type is wwr::ComplexToRealType<T>.
 *
 * SYNCHRONIZES @p stream. Both stages are host-driven, so on return the outputs
 * are complete and nothing is left queued -- except GebalJob::None, which only
 * sets the identity outputs asynchronously.
 *
 * Differences from a modern reference LAPACK (see README): magnitudes use the
 * off-diagonal 1-norm over the active window (EISPACK BALANC / LAPACK <= 3.4),
 * not the diagonal-inclusive 2-norm LAPACK 3.5+ adopted, so scale factors can
 * differ by a radix step; an index with non-finite norms is skipped rather than
 * reported as INFO = -3; and @p max_sweeps caps the otherwise-uncapped loop.
 *
 * Usage:
 *   import calaman.gebal;
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   int lwork = 0;
 *   calaman::gebal_bufferSize<double>(n, &lwork);     // ints of device scratch
 *   // d_A: n x n device matrix, lda; d_scale: length n; d_work: length lwork;
 *   int ilo = 0, ihi = 0;
 *   calaman::gebal(stream, calaman::GebalJob::Both, n, d_A, lda, &ilo, &ihi,
 *                  d_scale, d_work);
 */

module;

#include "gebal_bridge.h"

export module calaman.gebal;

import std;
import wwr.runtime_api;      // wwrStream_t, wwrError_t, wwrSuccess, wwrMemcpy/Memset/Sync
import wwr.complex;          // wwrFloatComplex, wwrDoubleComplex (extern template list)
import wwr.wrappers.common;  // usual_fp, ComplexToRealType

namespace calaman {

/// @brief Which halves of the balancing to run -- LAPACK's JOB argument
///
/// None    ('N'): no permutation, no scaling. ilo = 1, ihi = n, scale = 1.
/// Permute ('P'): isolate eigenvalues only; entry values are never changed.
/// Scale   ('S'): diagonal scaling only; ilo = 1, ihi = n.
/// Both    ('B'): permute, then scale.
export enum class GebalJob { None, Permute, Scale, Both };

/// @brief Default cap on scaling sweeps; see device::kGebalDefaultMaxSweeps
export inline constexpr int gebal_default_max_sweeps = device::kGebalDefaultMaxSweeps;

/// @brief Query the device scratch gebal needs, in ints (2n + 1)
///
/// @tparam T Element type (one of the instantiated types); the size is the same
///         for every T, but the template keeps the call symmetric with gebal.
/// @param n     Matrix dimension.
/// @param lwork Output: element count for the int workspace.
export template<wwr::usual_fp T>
void gebal_bufferSize(const int n, int *lwork) {
  *lwork = device::gebal_workspace_ints(n);
}

// A launch-error check must return the first failure, so it is a macro (an early
// `return` cannot be hidden behind a helper). Defined in the purview, used only
// by gebal's body, and #undef'd below; macros are never exported, so this does
// not leak to importers.
#define CLM_GEBAL_CHECK(expr)                                                                       \
  do {                                                                                              \
    const wwr::wwrError_t gebal_err_ = (expr);                                                      \
    if (gebal_err_ != wwr::wwrSuccess) {                                                            \
      return gebal_err_;                                                                            \
    }                                                                                               \
  } while (0)
#define CLM_GEBAL_CHECK_LAUNCH() CLM_GEBAL_CHECK(wwr::wwrGetLastError())

/// @brief Balance a general n-by-n matrix by a similarity transformation (?gebal)
///
/// Overwrites A with B = D^-1 P^T A P D, splitting the work exactly as LAPACK's
/// ?gebal does: first a symmetric permutation pushes every row whose off-diagonal
/// entries all vanish to the bottom (and every such column to the left), exposing
/// eigenvalues already on the diagonal; then the Parlett-Reinsch iteration picks
/// a powers-of-two diagonal D driving each row norm and its matching column norm
/// of the middle block B(ilo:ihi, ilo:ihi) towards each other. Powers of two make
/// the similarity exact in floating point, so undoing it reproduces A bit for bit.
///
/// Host-driven and SYNCHRONIZES @p stream (both stages branch on device data);
/// only GebalJob::None returns without a synchronization.
///
/// @tparam T Element type; one of the instantiated types (the four usual_fp types)
/// @param stream  Stream all kernels are launched on; A and the arrays live on its device
/// @param job     Which halves of the balancing to run
/// @param n       Matrix dimension
/// @param d_A     Device n-by-n matrix, column-major, overwritten with B
/// @param lda     Leading dimension of d_A (>= n)
/// @param ilo     Host output: 1-based first index of the balanced middle block
/// @param ihi     Host output: 1-based last index of it; rows ihi+1..n and columns
///                1..ilo-1 hold isolated eigenvalues on the diagonal
/// @param d_scale Device output, length n, real even for complex T. Inside
///                [ilo, ihi] it holds the diagonal of D; outside it holds the
///                1-based index the position was exchanged with. Both conventions
///                match LAPACK, so the array feeds ?gebak unchanged.
/// @param d_work  Device scratch of at least gebal_bufferSize<T>(n) ints
/// @param max_sweeps Safety cap on scaling sweeps
/// @return wwrSuccess, or the first runtime error encountered
export template<wwr::usual_fp T>
wwr::wwrError_t gebal(wwr::wwrStream_t stream, const GebalJob job, const int n, T *d_A,
                      const int lda, int *ilo, int *ihi, wwr::ComplexToRealType<T> *d_scale,
                      int *d_work, const int max_sweeps = device::kGebalDefaultMaxSweeps) {
  using R = wwr::ComplexToRealType<T>;

  const bool do_permute = (job == GebalJob::Permute) || (job == GebalJob::Both);
  const bool do_scale = (job == GebalJob::Scale) || (job == GebalJob::Both);

  if (n < 0 || lda < n || d_A == nullptr || d_scale == nullptr || ilo == nullptr ||
      ihi == nullptr) {
    return wwr::wwrErrorInvalidValue;
  }

  *ilo = 1;
  *ihi = n;
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  if (d_work == nullptr) {
    return wwr::wwrErrorInvalidValue;
  }

  // scale[] starts at 1 everywhere; the permutation stage overwrites only the
  // entries it isolates, which is what leaves 1s across [ilo, ihi].
  device::gebal_fill_ones<R>(stream, n, d_scale);
  CLM_GEBAL_CHECK_LAUNCH();

  if (!do_permute && !do_scale) {
    return wwr::wwrSuccess;
  }

  int *d_row_flag = d_work;
  int *d_col_flag = d_work + n;
  int *d_pick = d_work + 2 * n;
  const std::size_t flag_bytes = sizeof(int) * 2 * static_cast<std::size_t>(n);

  int k = 0;     // 0-based first index of the active window
  int l = n - 1; // 0-based last index of the active window

  if (do_permute) {
    // Push rows that isolate an eigenvalue to the bottom.
    for (;;) {
      if (l == 0) {
        // LAPACK returns the moment the window is down to one index.
        *ilo = 1;
        *ihi = 1;
        return wwr::wwrSuccess;
      }
      CLM_GEBAL_CHECK(wwr::wwrMemsetAsync(d_row_flag, 0, flag_bytes, stream));
      device::gebal_mark_nonzero<T>(stream, d_A, lda, 0, l, 0, l, d_row_flag, d_col_flag);
      CLM_GEBAL_CHECK_LAUNCH();

      device::gebal_set_int(stream, d_pick, -1);
      CLM_GEBAL_CHECK_LAUNCH();
      device::gebal_pick_max_unflagged(stream, d_row_flag, 0, l, d_pick);
      CLM_GEBAL_CHECK_LAUNCH();

      int j = -1;
      CLM_GEBAL_CHECK(
          wwr::wwrMemcpyAsync(&j, d_pick, sizeof(int), wwr::wwrMemcpyDeviceToHost, stream));
      CLM_GEBAL_CHECK(wwr::wwrStreamSynchronize(stream));
      if (j < 0) {
        break;
      }

      device::gebal_record_perm<R>(stream, d_scale, l, j + 1);
      CLM_GEBAL_CHECK_LAUNCH();
      if (j != l) {
        device::gebal_swap_cols<T>(stream, d_A, lda, n, j, l);
        CLM_GEBAL_CHECK_LAUNCH();
        device::gebal_swap_rows<T>(stream, d_A, lda, n, j, l);
        CLM_GEBAL_CHECK_LAUNCH();
      }
      --l;
    }

    // Push columns that isolate an eigenvalue to the left.
    while (k <= l) {
      CLM_GEBAL_CHECK(wwr::wwrMemsetAsync(d_row_flag, 0, flag_bytes, stream));
      device::gebal_mark_nonzero<T>(stream, d_A, lda, k, l, k, l, d_row_flag, d_col_flag);
      CLM_GEBAL_CHECK_LAUNCH();

      device::gebal_set_int(stream, d_pick, n);
      CLM_GEBAL_CHECK_LAUNCH();
      device::gebal_pick_min_unflagged(stream, d_col_flag, k, l, d_pick);
      CLM_GEBAL_CHECK_LAUNCH();

      int j = n;
      CLM_GEBAL_CHECK(
          wwr::wwrMemcpyAsync(&j, d_pick, sizeof(int), wwr::wwrMemcpyDeviceToHost, stream));
      CLM_GEBAL_CHECK(wwr::wwrStreamSynchronize(stream));
      if (j >= n) {
        break;
      }

      device::gebal_record_perm<R>(stream, d_scale, k, j + 1);
      CLM_GEBAL_CHECK_LAUNCH();
      if (j != k) {
        device::gebal_swap_cols<T>(stream, d_A, lda, n, j, k);
        CLM_GEBAL_CHECK_LAUNCH();
        device::gebal_swap_rows<T>(stream, d_A, lda, n, j, k);
        CLM_GEBAL_CHECK_LAUNCH();
      }
      ++k;
    }
  }

  *ilo = k + 1;
  *ihi = l + 1;

  if (do_scale && k <= l) {
    const device::scale_limits<R> lim = device::make_scale_limits<R>();
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
      CLM_GEBAL_CHECK(wwr::wwrMemsetAsync(d_pick, 0, sizeof(int), stream));
      device::gebal_sweep<T, R>(stream, n, d_A, lda, k, l, d_scale, d_pick, lim);
      CLM_GEBAL_CHECK_LAUNCH();

      int noconv = 0;
      CLM_GEBAL_CHECK(
          wwr::wwrMemcpyAsync(&noconv, d_pick, sizeof(int), wwr::wwrMemcpyDeviceToHost, stream));
      CLM_GEBAL_CHECK(wwr::wwrStreamSynchronize(stream));
      if (noconv == 0) {
        break;
      }
    }
  }

  return wwr::wwrSuccess;
}

#undef CLM_GEBAL_CHECK_LAUNCH
#undef CLM_GEBAL_CHECK

// Paired with instantiations.cpp: gebal is instantiated once inside this library
// (its body names the .cu-side launchers declared only in the GMF), so an
// importer never re-instantiates it. gebal_bufferSize is a trivial constexpr
// forward and is left to implicit instantiation.
extern template wwr::wwrError_t gebal<float>(wwr::wwrStream_t, GebalJob, int, float *, int, int *,
                                             int *, float *, int *, int);
extern template wwr::wwrError_t gebal<double>(wwr::wwrStream_t, GebalJob, int, double *, int, int *,
                                              int *, double *, int *, int);
extern template wwr::wwrError_t gebal<wwr::wwrFloatComplex>(wwr::wwrStream_t, GebalJob, int,
                                                           wwr::wwrFloatComplex *, int, int *, int *,
                                                           float *, int *, int);
extern template wwr::wwrError_t gebal<wwr::wwrDoubleComplex>(wwr::wwrStream_t, GebalJob, int,
                                                            wwr::wwrDoubleComplex *, int, int *,
                                                            int *, double *, int *, int);

} // namespace calaman
