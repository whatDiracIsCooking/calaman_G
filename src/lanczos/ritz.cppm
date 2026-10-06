/**
 * @file ritz.cppm
 * @brief Ritz extraction for the thick-restart Lanczos solver
 *
 * The :ritz partition of calaman.lanczos, in four composable stages:
 * lanczos_ritz_extract diagonalises T (syevd on a copy in s.s) and reads the
 * spectrum back in the cycle's one host sync; lanczos_ritz_select picks the
 * wanted pairs and their free residual estimates |beta_m s_{m,i}| on the host;
 * lanczos_ritz_compact moves the picked pairs to the leading slots of s.theta
 * and s.s (the layout lanczos_arrowhead reads); lanczos_ritz_vectors forms
 * X = V S_k with one gemm.
 *
 * A pair is converged when its estimate is at or below
 * tolerance * max(|theta_i|, ||T||_2). For k >= nev, the k-pair selection at
 * one LanczosWhich contains the nev-pair one.
 *
 * Usage:
 *   LanczosRitz<double> ritz;
 *   CLM_TRY(lanczos_ritz_extract<double>(solver, stream, ncv, s, &ritz));
 *   const auto sel = lanczos_ritz_select(ritz, which, nev, options.tolerance);
 *   CLM_TRY(lanczos_ritz_compact<double>(stream, ncv, sel.index, s));
 *   CLM_TRY(lanczos_ritz_vectors<double>(blas, n, ncv, nev, s, d_x, n));
 */

module;

// CLM_TRY / CLM_REQUIRE -- macros, so they arrive by #include in the GMF; they
// need calaman::Status visible at expansion, which the export import supplies.
#include "error_handling/error_macros.h"
#include "lanczos_bridge.h"

#include <cstddef>

export module calaman.lanczos:ritz;

import std;
import wwr.blas;            // wwrblasHandle_t, WWRBLAS_*
import wwr.solver;          // wwrsolverDnHandle_t, WWRSOLVER_EIG_MODE_VECTOR
import wwr.runtime_api;     // wwrMemcpyAsync, wwrMemcpy2DAsync, wwrStreamSynchronize
import wwr.wrappers.blas;   // gemm
import wwr.wrappers.solver; // syevd
import wwr.extension.blas;  // ScopedPointerMode
import :buffer_size;        // LanczosSlices
import :types;              // LanczosWhich
import calaman.common;      // kOne, kZero, real_fp
export import calaman.error_handling; // Status, PointerModeStatus

export namespace calaman {

/// @brief The host snapshot lanczos_ritz_extract reads back from one cycle.
template<calaman::real_fp T>
struct LanczosRitz {
  std::vector<T> theta;      ///< ncv Ritz values, ascending (s.theta)
  std::vector<T> s_last_row; ///< ncv: S(ncv-1, i), the last row of T's eigenvectors
  T beta_m = T{0};           ///< s.beta[ncv-1]: the coupling to v_ncv
  T t_norm = T{0};           ///< ||T||_2 = max_i |theta_i|
  bool breakdown = false;    ///< the cycle's status block: a step tripped the guard
  int breakdown_step = -1;   ///< its first such step, -1 if none
};

/// @brief The wanted pairs out of a LanczosRitz, in ascending theta order.
template<calaman::real_fp T>
struct LanczosRitzSelection {
  std::vector<int> index;      ///< positions in LanczosRitz::theta, strictly ascending
  std::vector<T> values;       ///< theta at index
  std::vector<T> residuals;    ///< |beta_m * s_last_row| at index
  std::vector<bool> converged; ///< residuals[i] <= tol * max(|values[i]|, t_norm)
  int converged_count = 0;

  bool all_converged() const { return converged_count == static_cast<int>(index.size()); }
};

/// @brief The @p count positions, ascending, of an ascending ncv-spectrum that
///        @p which wants (both_ends: ceil(count/2) top, floor(count/2) bottom).
/// @return Empty unless 1 <= count <= ncv.
inline std::vector<int> lanczos_select(const LanczosWhich which, const int ncv, const int count) {
  std::vector<int> index;
  if (count < 1 || count > ncv) {
    return index;
  }
  int low = 0; // how many come from the bottom; the rest from the top
  switch (which) {
  case LanczosWhich::smallest:
    low = count;
    break;
  case LanczosWhich::largest:
    low = 0;
    break;
  case LanczosWhich::both_ends:
    low = count / 2;
    break;
  }
  index.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < low; ++i) {
    index.push_back(i);
  }
  for (int i = ncv - (count - low); i < ncv; ++i) {
    index.push_back(i);
  }
  return index;
}

/// @brief Select @p count pairs of @p ritz at @p which, with their residual
///        estimates and convergence flags. Host only; empty on a bad count.
template<calaman::real_fp T>
LanczosRitzSelection<T> lanczos_ritz_select(const LanczosRitz<T> &ritz, const LanczosWhich which,
                                            const int count, const T tolerance) {
  LanczosRitzSelection<T> sel;
  sel.index = lanczos_select(which, static_cast<int>(ritz.theta.size()), count);
  for (const int i : sel.index) {
    const auto iz = static_cast<std::size_t>(i);
    const T theta = ritz.theta[iz];
    const T residual = std::abs(ritz.beta_m * ritz.s_last_row[iz]);
    const bool ok = residual <= tolerance * std::max(std::abs(theta), ritz.t_norm);
    sel.values.push_back(theta);
    sel.residuals.push_back(residual);
    sel.converged.push_back(ok);
    sel.converged_count += ok ? 1 : 0;
  }
  return sel;
}

/**
 * @brief Diagonalise the projected matrix: s.s = eigenvectors of s.t (copied,
 *        so s.t survives), s.theta = eigenvalues ascending; then read the
 *        spectrum, S's last row, beta_m and the status block back in ONE sync.
 *
 * @param ritz Out (host). Filled before a nonzero syevd devInfo is reported.
 * @return INVALID_VALUE for a null @p ritz or ncv < 1; INTERNAL_ERROR when
 *         syevd's devInfo is nonzero; otherwise the first failing call's status.
 */
template<calaman::real_fp T>
Status lanczos_ritz_extract(wwr::wwrsolverDnHandle_t solver_handle, wwr::wwrStream_t stream,
                            const int ncv, const LanczosSlices<T> &s, LanczosRitz<T> *ritz) {
  CLM_REQUIRE(ritz != nullptr && ncv >= 1, wwr::WWRBLAS_STATUS_INVALID_VALUE);
  const auto mz = static_cast<std::size_t>(ncv);

  CLM_TRY(wwr::wwrMemcpyAsync(s.s, s.t, sizeof(T) * mz * mz, wwr::wwrMemcpyDeviceToDevice,
                              stream));
  CLM_TRY(wwr::syevd<T>(solver_handle, wwr::WWRSOLVER_EIG_MODE_VECTOR,
                        wwr::WWRBLAS_FILL_MODE_LOWER, ncv, s.s, ncv, s.theta, s.eig_scratch,
                        s.lwork_eig, s.eig_info));

  ritz->theta.resize(mz);
  ritz->s_last_row.resize(mz);
  device::LanczosStatus status{};
  CLM_TRY(wwr::wwrMemcpyAsync(ritz->theta.data(), s.theta, sizeof(T) * mz,
                              wwr::wwrMemcpyDeviceToHost, stream));
  // Row ncv-1 of the column-major S: one element per column, pitch ncv.
  CLM_TRY(wwr::wwrMemcpy2DAsync(ritz->s_last_row.data(), sizeof(T), s.s + (mz - 1),
                                sizeof(T) * mz, sizeof(T), mz, wwr::wwrMemcpyDeviceToHost,
                                stream));
  CLM_TRY(wwr::wwrMemcpyAsync(&ritz->beta_m, s.beta + (mz - 1), sizeof(T),
                              wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(&status, s.status, sizeof(status), wwr::wwrMemcpyDeviceToHost,
                              stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));

  ritz->t_norm = std::max(std::abs(ritz->theta.front()), std::abs(ritz->theta.back()));
  ritz->breakdown = status.breakdown != 0;
  ritz->breakdown_step = status.breakdown_step;
  // A host-side verdict on devInfo carries a BLAS-domain code, as in davidson.
  CLM_REQUIRE(status.eig_info == 0, wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief Move the pairs at @p index to the leading slots: s.theta[j] =
 *        s.theta[index[j]] and S(:, j) = S(:, index[j]). Device copies, no sync.
 *
 * @param index Strictly ascending positions in [0, ncv) (a selection's index).
 * @return INVALID_VALUE for an index that is not.
 */
template<calaman::real_fp T>
Status lanczos_ritz_compact(wwr::wwrStream_t stream, const int ncv, const std::vector<int> &index,
                            const LanczosSlices<T> &s) {
  for (std::size_t j = 0; j < index.size(); ++j) {
    CLM_REQUIRE(index[j] >= 0 && index[j] < ncv && (j == 0 || index[j] > index[j - 1]),
                wwr::WWRBLAS_STATUS_INVALID_VALUE);
  }
  const auto mz = static_cast<std::size_t>(ncv);
  // index[j] >= j, so each source is read before any copy lands on it. A run of
  // equal shift g moves in chunks of at most g columns, keeping every single
  // memcpy's source and destination disjoint.
  std::size_t j = 0;
  while (j < index.size()) {
    const std::size_t shift = static_cast<std::size_t>(index[j]) - j;
    std::size_t run = 1;
    while (j + run < index.size() && static_cast<std::size_t>(index[j + run]) - (j + run) == shift) {
      ++run;
    }
    for (std::size_t c = 0; shift != 0 && c < run; c += shift) {
      const std::size_t dst = j + c;
      const std::size_t width = std::min(shift, run - c);
      CLM_TRY(wwr::wwrMemcpyAsync(s.theta + dst, s.theta + dst + shift, sizeof(T) * width,
                                  wwr::wwrMemcpyDeviceToDevice, stream));
      CLM_TRY(wwr::wwrMemcpyAsync(s.s + dst * mz, s.s + (dst + shift) * mz,
                                  sizeof(T) * mz * width, wwr::wwrMemcpyDeviceToDevice, stream));
    }
    j += run;
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief Ritz vectors X = V(:, 0:ncv) * S(:, 0:count): the leading @p count
 *        (compacted) pairs rotated out of the basis, by one gemm in HOST
 *        pointer mode (the handle's mode is restored on return).
 *
 * @param x   Out: n x count device, ld @p ldx >= n; must not overlap V.
 * @return INVALID_VALUE for a bad count/ldx or a null @p x.
 */
template<calaman::real_fp T>
Status lanczos_ritz_vectors(wwr::wwrblasHandle_t blas_handle, const int n, const int ncv,
                            const int count, const LanczosSlices<T> &s, T *x, const int ldx) {
  CLM_REQUIRE(x != nullptr && count >= 1 && count <= ncv && ldx >= n,
              wwr::WWRBLAS_STATUS_INVALID_VALUE);
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode mode{blas_handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                               PointerModeStatus{&pm_status}};
  CLM_TRY(pm_status);
  CLM_TRY((wwr::gemm<T, int>(blas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, count, ncv,
                             &kOne<T>, s.v, n, s.s, ncv, &kZero<T>, x, ldx)));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
