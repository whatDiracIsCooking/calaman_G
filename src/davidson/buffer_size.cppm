/**
 * @file buffer_size.cppm
 * @brief Device workspace layout and sizing for the block-Davidson solver
 *
 * The :buffer_size partition of calaman.davidson.
 *
 * One caller-provided device buffer, laid out by the single DavidsonSlices::carve
 * through calaman::carve_workspace (null base sizes, real base carves), so the
 * size query and the carving can never drift. Every region starts 256-byte
 * aligned. davidson_solve therefore allocates no device memory, and a repeat
 * solve at the same (n, n_roots, max_subspace) reuses the same buffer.
 *
 * The subspace basis V and its operator image Sigma_V dominate: each is
 * n x max_subspace. Everything else is O(n n_roots) or O(max_subspace^2). The
 * syevd (Euclidean) and sygvd (metric) eigensolver workspaces are never live at
 * the same time, so they share one SCRATCH region sized to the larger.
 *
 * The metric (generalized) regions -- M V, V^T M V and the M-image of the
 * residual/correction block -- are carved ONLY when with_metric is set, so an
 * Euclidean sizing pays nothing for the path it will not take.
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

#include <cstddef>

export module calaman.davidson:buffer_size;

import std;
import wwr.blas;            // WWRBLAS_STATUS_*, wwrblasFillMode_t, WWRBLAS_FILL_MODE_*
import wwr.solver;          // wwrsolverDnHandle_t, wwrsolverEig*_t, WWRSOLVER_EIG_*
import wwr.wrappers.solver; // syevd_bufferSize, sygvd_bufferSize
import calaman.common;      // WorkspaceLayout, carve_workspace, real_fp
export import calaman.error_handling; // Status -- the cross-domain return type

export namespace calaman {

/**
 * @brief Pointers into the solver's single device workspace buffer.
 *
 * Built once by make_davidson_slices and passed to davidson_solve, so the solve
 * allocates nothing. The metric_* members are null on the Euclidean path (the
 * workspace was sized without them) and non-null only when the slices were made
 * with_metric.
 */
template<calaman::real_fp T>
struct DavidsonSlices {
  // ── Euclidean path (always carved) ──────────────────────────────────────
  T *v = nullptr;          ///< n x max_subspace: the subspace basis, ld n
  T *av = nullptr;         ///< n x max_subspace: Sigma_V = A V, kept in lockstep with v
  T *h = nullptr;          ///< max_subspace x max_subspace: subspace matrix, then its eigenvectors
  T *ritz = nullptr;       ///< max_subspace: subspace eigenvalues, ascending
  T *ritz_av = nullptr;    ///< n x n_roots: A X for the current Ritz vectors
  T *residual = nullptr;   ///< n x n_roots: A X - X diag(theta)
  T *correction = nullptr; ///< n x n_roots: preconditioner output, then Gram-Schmidt scratch
  T *proj = nullptr;       ///< max_subspace x n_roots: V^T correction, ld max_subspace
  int *info = nullptr;     ///< one int: the syevd/sygvd devInfo
  int lwork_eig = 0;       ///< eig_scratch length, in elements of T

  T *eig_scratch = nullptr; ///< syevd (and, on the metric path, sygvd) device workspace

  // ── metric (generalized) path only; null unless made with_metric ────────
  T *mv = nullptr;             ///< n x max_subspace: M V, kept in lockstep with v
  T *s_sub = nullptr;          ///< max_subspace x max_subspace: V^T M V (SPD overlap)
  T *metric_scratch = nullptr; ///< n x n_roots: M applied to the residual/correction block

  /// @brief Lay the slices out from @p layout -- the ONLY description of the
  ///        layout, run for sizing and carving alike (see carve_workspace).
  ///        All regions are FIXED except the single shared eigensolver SCRATCH
  ///        block, carved last. @p eig_len is syevd's lwork, or the syevd/sygvd
  ///        max on the metric path.
  void carve(WorkspaceLayout &layout, const int n, const int n_roots, const int max_subspace,
             const bool with_metric, const int eig_len) {
    const std::size_t nz = static_cast<std::size_t>(n);
    const std::size_t rz = static_cast<std::size_t>(n_roots);
    const std::size_t mz = static_cast<std::size_t>(max_subspace);

    lwork_eig = eig_len;

    v = layout.fixed<T>(nz * mz);
    av = layout.fixed<T>(nz * mz);
    h = layout.fixed<T>(mz * mz);
    ritz = layout.fixed<T>(mz);
    ritz_av = layout.fixed<T>(nz * rz);
    residual = layout.fixed<T>(nz * rz);
    correction = layout.fixed<T>(nz * rz);
    proj = layout.fixed<T>(mz * rz);
    info = layout.fixed<int>(1);

    if (with_metric) {
      mv = layout.fixed<T>(nz * mz);
      s_sub = layout.fixed<T>(mz * mz);
      metric_scratch = layout.fixed<T>(nz * rz);
    }

    // One SCRATCH region: syevd's workspace on the Euclidean path, sygvd's on
    // the metric path. They are never live together, so the block is the larger
    // of the two (eig_len already holds that max).
    eig_scratch = layout.scratch<T>(static_cast<std::size_t>(eig_len));
  }
};

} // namespace calaman

export namespace calaman {

/**
 * @brief Carve @p d_work into DavidsonSlices for an n-dimensional operator's
 *        lowest @p n_roots eigenpairs, with a subspace bounded at @p max_subspace.
 *
 * Validates the shape (the same contract create() enforced in the origin
 * project: positive sizes, n_roots <= n, 2 * n_roots <= max_subspace <= n),
 * queries the eigensolver workspace at the largest subspace it will diagonalize,
 * and carves. Pass @p with_metric to size the generalized (sygvd) path too.
 *
 * @param d_work      Workspace, or null to size it only.
 * @param slices      Out, may be null.
 * @param lwork_bytes Out, may be null: the bytes the layout needs.
 */
template<calaman::real_fp T>
Status make_davidson_slices(wwr::wwrsolverDnHandle_t cusolver_handle, const int n, const int n_roots,
                            const int max_subspace, const bool with_metric, void *d_work,
                            DavidsonSlices<T> *slices, std::size_t *lwork_bytes) {
  CLM_REQUIRE(n > 0 && n_roots > 0, wwr::WWRBLAS_STATUS_INVALID_VALUE);
  CLM_REQUIRE(n_roots <= n, wwr::WWRBLAS_STATUS_INVALID_VALUE);
  CLM_REQUIRE(max_subspace >= 2 * n_roots, wwr::WWRBLAS_STATUS_INVALID_VALUE);
  CLM_REQUIRE(max_subspace <= n, wwr::WWRBLAS_STATUS_INVALID_VALUE);

  int lwork_eig = 0;
  CLM_TRY(wwr::syevd_bufferSize<T>(cusolver_handle, wwr::WWRSOLVER_EIG_MODE_VECTOR,
                                   wwr::WWRBLAS_FILL_MODE_LOWER, max_subspace,
                                   static_cast<T *>(nullptr), max_subspace,
                                   static_cast<T *>(nullptr), &lwork_eig));

  if (with_metric) {
    int lwork_sygvd = 0;
    CLM_TRY(wwr::sygvd_bufferSize<T>(
        cusolver_handle, wwr::WWRSOLVER_EIG_TYPE_1, wwr::WWRSOLVER_EIG_MODE_VECTOR,
        wwr::WWRBLAS_FILL_MODE_LOWER, max_subspace, static_cast<T *>(nullptr), max_subspace,
        static_cast<T *>(nullptr), max_subspace, static_cast<T *>(nullptr), &lwork_sygvd));
    lwork_eig = std::max(lwork_eig, lwork_sygvd);
  }

  const std::size_t bytes =
      carve_workspace(d_work, slices, n, n_roots, max_subspace, with_metric, lwork_eig);
  if (lwork_bytes != nullptr) {
    *lwork_bytes = bytes;
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief Device workspace, in bytes, that davidson_solve needs.
 *
 * @param with_metric  Size the generalized (sygvd) path too; must match the
 *                     value passed to make_davidson_slices and davidson_solve.
 * @param lwork_bytes  Out: bytes required.
 */
template<calaman::real_fp T>
Status davidson_bufferSize(wwr::wwrsolverDnHandle_t cusolver_handle, const int n, const int n_roots,
                           const int max_subspace, const bool with_metric,
                           std::size_t *lwork_bytes) {
  CLM_REQUIRE(lwork_bytes != nullptr, wwr::WWRBLAS_STATUS_INVALID_VALUE);
  return make_davidson_slices<T>(cusolver_handle, n, n_roots, max_subspace, with_metric, nullptr,
                                 nullptr, lwork_bytes);
}

} // namespace calaman
