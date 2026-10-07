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
 * Selection, classification and the rotation are calaman.ritz's (ritz_select,
 * classify_ritz with scale ||T||_2, ritz_rotate); this partition adapts them to
 * LanczosSlices and keeps what is Lanczos-only.
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
import wwr.wrappers.solver; // syevd
import :buffer_size;        // LanczosSlices
import :types;              // LanczosWhich
import calaman.common;      // real_fp
export import calaman.ritz; // RitzSelection, ritz_select, classify_ritz, ritz_rotate
export import calaman.error_handling; // Status

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

/// @brief Select @p count pairs of @p ritz at @p which (ritz_select), with the
///        estimates |beta_m s_{m,i}| classified at scale ||T||_2 (classify_ritz).
///        index holds the positions in ritz.theta. Host only; empty on a bad count.
template<calaman::real_fp T>
RitzSelection<T> lanczos_ritz_select(const LanczosRitz<T> &ritz, const LanczosWhich which,
                                     const int count, const T tolerance) {
  std::vector<int> index = ritz_select(which, static_cast<int>(ritz.theta.size()), count);
  std::vector<T> values;
  std::vector<T> residuals;
  values.reserve(index.size());
  residuals.reserve(index.size());
  for (const int i : index) {
    const auto iz = static_cast<std::size_t>(i);
    values.push_back(ritz.theta[iz]);
    residuals.push_back(std::abs(ritz.beta_m * ritz.s_last_row[iz]));
  }
  RitzSelection<T> sel = classify_ritz<T>(values, residuals, tolerance, ritz.t_norm);
  sel.index = std::move(index);
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
  // devInfo's domain and code: calaman::devinfo_verdict (calaman.error_handling).
  return devinfo_verdict(status.eig_info);
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
 *        (compacted) pairs rotated out of the basis -- ritz_rotate on s.v, s.s.
 *
 * @param x   Out: n x count device, ld @p ldx >= n; must not overlap V.
 * @return INVALID_VALUE for a bad count/ldx or a null @p x.
 */
template<calaman::real_fp T>
Status lanczos_ritz_vectors(wwr::wwrblasHandle_t blas_handle, const int n, const int ncv,
                            const int count, const LanczosSlices<T> &s, T *x, const int ldx) {
  return ritz_rotate<T>(blas_handle, n, ncv, count, s.v, n, s.s, ncv, x, ldx);
}

} // namespace calaman
