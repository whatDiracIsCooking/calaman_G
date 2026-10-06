/**
 * @file buffer_size.cppm
 * @brief Device workspace layout and sizing for the thick-restart Lanczos solver
 *
 * The :buffer_size partition of calaman.lanczos. One caller-provided device
 * buffer, laid out by the single LanczosSlices::carve through
 * calaman::carve_workspace (null base sizes, real base carves), so the size query
 * and the carving cannot drift. Every region starts 256-byte aligned.
 *
 * The basis V (n x (ncv + 1)) and the restart's staging block (n x k, k =
 * lanczos_restart_keep) dominate; the rest is O(ncv^2) plus one generator
 * state per row for the random start / breakdown-recovery vectors. The syevd
 * workspace is the single SCRATCH region.
 *
 * Shape contract: nev >= 1, 2 * nev + 1 <= ncv <= n.
 */

module;

// CLM_TRY / CLM_REQUIRE -- macros, so they arrive by #include in the GMF; they
// need calaman::Status visible at expansion, which the export import supplies.
#include "error_handling/error_macros.h"
#include "lanczos_bridge.h"

#include <cstddef>

export module calaman.lanczos:buffer_size;

import std;
import wwr.blas;            // WWRBLAS_STATUS_*, WWRBLAS_FILL_MODE_*
import wwr.solver;          // wwrsolverDnHandle_t, WWRSOLVER_EIG_MODE_VECTOR
import wwr.rand;            // wwrrandState
import wwr.wrappers.solver; // syevd_bufferSize
import calaman.common;      // WorkspaceLayout, member_ptr, carve_workspace, slices_for, real_fp
export import calaman.error_handling; // Status -- the cross-domain return type

export namespace calaman {

/// @brief Ritz pairs a thick restart keeps: nev + (ncv - nev) / 2, so each later
///        cycle adds about as many new steps as it keeps. nev <= k <= ncv - 1
///        whenever lanczos_shape_ok.
constexpr int lanczos_restart_keep(const int nev, const int ncv) noexcept {
  return nev + (ncv - nev) / 2;
}

/// @brief Pointers into the solver's single device workspace buffer.
///
/// Built once by make_lanczos_slices and passed to lanczos_solve, so the solve
/// allocates nothing. All matrices are column-major.
template<calaman::real_fp T>
struct LanczosSlices {
  T *v = nullptr;            ///< n x (ncv + 1): the Lanczos basis, ld n
  T *keep = nullptr;         ///< n x k: V S_k, the kept Ritz vectors staged for a restart
  T *t = nullptr;            ///< ncv x ncv: the projected matrix T, ld ncv
  T *s = nullptr;            ///< ncv x ncv: T's eigenvectors (syevd output), ld ncv
  T *theta = nullptr;        ///< ncv: Ritz values, ascending
  T *coeffs = nullptr;       ///< ncv: CGS2 reorthogonalization coefficients V^T w
  T *alpha = nullptr;        ///< ncv: alpha_j history (T's diagonal)
  T *beta = nullptr;         ///< ncv: beta_j history (T's subdiagonal / coupling)
  wwr::wwrrandState *rng = nullptr; ///< n generator states: random start / restart vectors
  device::LanczosStatus *status = nullptr; ///< per-cycle status block
  int *eig_info = nullptr;   ///< &status->eig_info: syevd's devInfo
  int lwork_eig = 0;         ///< eig_scratch length, in elements of T

  T *eig_scratch = nullptr;  ///< syevd device workspace (the one SCRATCH region)

  /// @brief Lay the slices out from @p layout -- the ONLY description of the
  ///        layout, run for sizing and carving alike. Every region is FIXED
  ///        except eig_scratch, carved last. @p eig_len is syevd's lwork at ncv.
  void carve(WorkspaceLayout &layout, const int n, const int nev, const int ncv,
             const int eig_len) {
    const std::size_t nz = static_cast<std::size_t>(n);
    const std::size_t mz = static_cast<std::size_t>(ncv);
    const std::size_t kz = static_cast<std::size_t>(lanczos_restart_keep(nev, ncv));

    lwork_eig = eig_len;

    v = layout.fixed<T>(nz * (mz + 1));
    keep = layout.fixed<T>(nz * kz);
    t = layout.fixed<T>(mz * mz);
    s = layout.fixed<T>(mz * mz);
    theta = layout.fixed<T>(mz);
    coeffs = layout.fixed<T>(mz);
    alpha = layout.fixed<T>(mz);
    beta = layout.fixed<T>(mz);
    rng = layout.fixed<wwr::wwrrandState>(nz);
    status = layout.fixed_struct<device::LanczosStatus>();
    eig_info = member_ptr(status, &device::LanczosStatus::eig_info);

    eig_scratch = layout.scratch<T>(static_cast<std::size_t>(eig_len));
  }
};

/// @brief Whether (n, nev, ncv) is a shape the solver accepts:
///        nev >= 1 and 2 * nev + 1 <= ncv <= n.
constexpr bool lanczos_shape_ok(const int n, const int nev, const int ncv) noexcept {
  return nev >= 1 && ncv >= 2 * nev + 1 && ncv <= n;
}

/**
 * @brief Carve @p d_work into LanczosSlices for the @p nev extreme eigenpairs of
 *        an n-dimensional operator, with a basis of @p ncv vectors per cycle.
 *
 * @param d_work      Workspace, or null to size it only.
 * @param slices      Out, may be null.
 * @param lwork_bytes Out, may be null: the bytes the layout needs.
 * @return INVALID_VALUE unless lanczos_shape_ok(n, nev, ncv).
 */
template<calaman::real_fp T>
Status make_lanczos_slices(wwr::wwrsolverDnHandle_t solver_handle, const int n, const int nev,
                           const int ncv, void *d_work, LanczosSlices<T> *slices,
                           std::size_t *lwork_bytes) {
  CLM_REQUIRE(lanczos_shape_ok(n, nev, ncv), wwr::WWRBLAS_STATUS_INVALID_VALUE);

  int lwork_eig = 0;
  CLM_TRY(wwr::syevd_bufferSize<T>(solver_handle, wwr::WWRSOLVER_EIG_MODE_VECTOR,
                                   wwr::WWRBLAS_FILL_MODE_LOWER, ncv, static_cast<T *>(nullptr),
                                   ncv, static_cast<T *>(nullptr), &lwork_eig));

  const std::size_t bytes = carve_workspace(d_work, slices, n, nev, ncv, lwork_eig);
  if (lwork_bytes != nullptr) {
    *lwork_bytes = bytes;
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief Device workspace, in bytes, that lanczos_solve needs at (n, nev, ncv).
/// @return INVALID_VALUE for a null @p lwork_bytes or a rejected shape.
template<calaman::real_fp T>
Status lanczos_bufferSize(wwr::wwrsolverDnHandle_t solver_handle, const int n, const int nev,
                          const int ncv, std::size_t *lwork_bytes) {
  CLM_REQUIRE(lwork_bytes != nullptr, wwr::WWRBLAS_STATUS_INVALID_VALUE);
  return make_lanczos_slices<T>(solver_handle, n, nev, ncv, nullptr, nullptr, lwork_bytes);
}

} // namespace calaman

namespace calaman {
static_assert(slices_for<LanczosSlices<float>, int, int, int, int>);
static_assert(slices_for<LanczosSlices<double>, int, int, int, int>);
} // namespace calaman
