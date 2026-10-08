/**
 * @file buffer_size.cppm
 * @brief Device workspace layout and sizing for the FEAST solver
 *
 * The :buffer_size partition of calaman.feast.
 *
 * FeastSlices is the driver's own O(n m0) part: the Rayleigh-Ritz blocks and
 * the per-iteration status. A resolvent model carves its own workspace beside
 * it; DenseFeastWorkspace composes the two for the dense feast entry point, so
 * one caller buffer is laid out by one carve through calaman::carve_workspace
 * (null base sizes, real base carves) and the size query and the carving
 * cannot drift apart. Every region starts 256-byte aligned.
 *
 * The dense model's Ne resolvents dominate: Ne n^2 complex elements, 32 MiB at
 * n = 512 in double with Ne = 8. The QR and eigensolver workspaces are never
 * live at the same time, so they share the layout's one SCRATCH region.
 *
 * make_feast_slices / feast_bufferSize return a @ref calaman::Status: an
 * argument fault is a BLAS INVALID_VALUE, and a failing solver lwork query
 * flows through CLM_TRY in its own solver domain.
 */

module;

#include "feast_bridge.h"

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.feast:buffer_size;

import std;
import wwr.blas;            // WWRBLAS_STATUS_*, wwrblasFillMode_t, WWRBLAS_FILL_MODE_*
import wwr.solver;          // wwrsolverDnHandle_t, wwrsolverEigMode_t, WWRSOLVER_EIG_MODE_VECTOR
import wwr.wrappers.solver; // syevd_bufferSize
import calaman.common;      // align_up, WorkspaceLayout, member_ptr, carve_workspace, real_fp, ...
import calaman.orthogonalize;         // orthogonalize_bufferSize
import calaman.lacn2;                 // lacn2_bufferSize
import :resolvent;                    // DenseResolventSlices
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

/**
 * @brief Pointers into the driver's own workspace: Rayleigh-Ritz and status.
 *
 * Built once per solve and passed to each step, so the steps allocate nothing.
 */
template<calaman::real_fp T>
struct FeastSlices {
  // ── Rayleigh-Ritz ──────────────────────────────────────────────────────
  T *basis = nullptr;     ///< n x m0: the filtered subspace, orthonormalized in place
  T *a_basis = nullptr;   ///< n x m0: A * basis
  T *a_ritz = nullptr;    ///< n x m0: A * the Ritz vectors, for the residuals
  T *projected = nullptr; ///< m0 x m0: basis^T A basis, then its eigenvectors
  T *rotated = nullptr;   ///< m0 x m0: those eigenvectors, in-interval columns first
  T *ritz = nullptr;      ///< m0: the Ritz values, ascending, as syevd returns them
  T *residuals = nullptr; ///< m0: relative residual of each returned pair; 0 past status->m
  T *norm_a = nullptr;    ///< one element: ||A||_1, or lacn2's lower bound on it

  // ── the per-iteration status, and pointers into it ─────────────────────
  device::FeastStatus<T> *status = nullptr;
  int *qr_info = nullptr;  ///< status->qr_info
  int *eig_info = nullptr; ///< &status->eig_info

  // ── shared solver scratch ──────────────────────────────────────────────
  T *scratch = nullptr; ///< orthogonalize's workspace, then syevd's
  int lwork_qr = 0;     ///< orthogonalize's lwork, in elements of T
  int lwork_eig = 0;    ///< syevd's lwork, in elements of T
  void *norm_work = nullptr;      ///< lacn2's workspace, aliasing scratch: used before the loop
  std::size_t norm_work_bytes = 0; ///< its size, lacn2_bufferSize<T>(n)

  /// @brief Lay the slices out from @p layout: FIXED regions, then the one
  ///        shared QR/eigensolver/lacn2 SCRATCH block. @p qr_len / @p eig_len
  ///        are the orthogonalize / syevd lworks, in elements of T.
  void carve(WorkspaceLayout &layout, const int n, const int m0, const int qr_len,
             const int eig_len) {
    const std::size_t nz = static_cast<std::size_t>(n);
    const std::size_t m0z = static_cast<std::size_t>(m0);

    lwork_qr = qr_len;
    lwork_eig = eig_len;

    basis = layout.fixed<T>(nz * m0z);
    a_basis = layout.fixed<T>(nz * m0z);
    a_ritz = layout.fixed<T>(nz * m0z);
    projected = layout.fixed<T>(m0z * m0z);
    rotated = layout.fixed<T>(m0z * m0z);
    ritz = layout.fixed<T>(m0z);
    residuals = layout.fixed<T>(m0z);
    norm_a = layout.fixed<T>(1);

    status = layout.fixed_struct<device::FeastStatus<T>>();
    qr_info = member_ptr(status, &device::FeastStatus<T>::qr_info);
    eig_info = member_ptr(status, &device::FeastStatus<T>::eig_info);

    // One SCRATCH region: orthogonalize's workspace, then syevd's. They are
    // never live together, so the block is sized to the larger and reused.
    scratch = layout.scratch<T>(static_cast<std::size_t>(std::max(qr_len, eig_len)));
    // lacn2's ||A||_1 estimate runs once, before either, so it aliases them too.
    norm_work_bytes = lacn2_bufferSize<T>(n);
    norm_work = layout.scratch<std::byte>(norm_work_bytes);
  }
};

/// @brief The dense feast entry point's one buffer: the DenseResolvent model's
///        regions (all FIXED) first, then the driver's FeastSlices.
template<calaman::real_fp T>
struct DenseFeastWorkspace {
  DenseResolventSlices<T> resolvent;
  FeastSlices<T> rr;

  void carve(WorkspaceLayout &layout, const int n, const int m0, const int ne, const int qr_len,
             const int eig_len) {
    resolvent.carve(layout, n, m0, ne);
    rr.carve(layout, n, m0, qr_len, eig_len);
  }
};

} // namespace calaman

namespace calaman {

/// @brief The orthogonalize and syevd lworks, in elements of T, at (n, m0).
template<calaman::real_fp T>
Status feast_solver_lworks(wwr::wwrsolverDnHandle_t cusolver_handle, const int n, const int m0,
                           int &lwork_qr, int &lwork_eig) {
  CLM_TRY(orthogonalize_bufferSize<T>(cusolver_handle, n, m0, &lwork_qr));
  CLM_TRY(wwr::syevd_bufferSize<T>(cusolver_handle, wwr::WWRSOLVER_EIG_MODE_VECTOR,
                                   wwr::WWRBLAS_FILL_MODE_LOWER, m0, static_cast<T *>(nullptr), m0,
                                   static_cast<T *>(nullptr), &lwork_eig));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief Carve @p d_work into a DenseFeastWorkspace for an n x n problem with an
 *        m0-column subspace and Ne quadrature nodes.
 *
 * @param d_work      Workspace, or null to size it only.
 * @param slices      Out, may be null.
 * @param lwork_bytes Out, may be null: the bytes the layout needs.
 */
template<calaman::real_fp T, std::size_t Ne>
  requires(Ne == 4 || Ne == 8)
Status make_feast_slices(wwr::wwrsolverDnHandle_t cusolver_handle, const int n, const int m0,
                         void *d_work, DenseFeastWorkspace<T> *slices, std::size_t *lwork_bytes) {
  if (n < 1 || m0 < 1 || m0 > n) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  int lwork_qr = 0;
  int lwork_eig = 0;
  CLM_TRY(feast_solver_lworks<T>(cusolver_handle, n, m0, lwork_qr, lwork_eig));

  const std::size_t bytes =
      carve_workspace(d_work, slices, n, m0, static_cast<int>(Ne), lwork_qr, lwork_eig);
  if (lwork_bytes != nullptr) {
    *lwork_bytes = bytes;
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief Carve @p d_work into the driver's FeastSlices alone, for a caller-built
///        model that carves its own workspace. Arguments as make_feast_slices.
template<calaman::real_fp T>
Status make_feast_driver_slices(wwr::wwrsolverDnHandle_t cusolver_handle, const int n,
                                const int m0, void *d_work, FeastSlices<T> *slices,
                                std::size_t *lwork_bytes) {
  if (n < 1 || m0 < 1 || m0 > n) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  int lwork_qr = 0;
  int lwork_eig = 0;
  CLM_TRY(feast_solver_lworks<T>(cusolver_handle, n, m0, lwork_qr, lwork_eig));

  const std::size_t bytes = carve_workspace(d_work, slices, n, m0, lwork_qr, lwork_eig);
  if (lwork_bytes != nullptr) {
    *lwork_bytes = bytes;
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief Device workspace, in bytes, that feast needs.
 *
 * @tparam T  Real floating-point type (float or double).
 * @tparam Ne Number of quadrature nodes (4 or 8); must match feast's.
 * @param n           Matrix dimension.
 * @param m0          Subspace size, 1 <= m0 <= n.
 * @param lwork_bytes Out: bytes required.
 */
export template<calaman::real_fp T, std::size_t Ne = 8>
  requires(Ne == 4 || Ne == 8)
Status feast_bufferSize(wwr::wwrsolverDnHandle_t cusolver_handle, const int n, const int m0,
                        std::size_t *lwork_bytes) {
  if (lwork_bytes == nullptr) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  return make_feast_slices<T, Ne>(cusolver_handle, n, m0, nullptr, nullptr, lwork_bytes);
}

/// @brief Device workspace, in bytes, that feast over a caller's model needs:
///        the driver's own O(n m0) part. The model's is sized separately.
export template<calaman::real_fp T>
Status feast_driver_bufferSize(wwr::wwrsolverDnHandle_t cusolver_handle, const int n,
                               const int m0, std::size_t *lwork_bytes) {
  if (lwork_bytes == nullptr) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  return make_feast_driver_slices<T>(cusolver_handle, n, m0, nullptr, nullptr, lwork_bytes);
}

} // namespace calaman
