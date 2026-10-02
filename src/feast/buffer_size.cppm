/**
 * @file buffer_size.cppm
 * @brief Device workspace layout and sizing for the FEAST solver
 *
 * The :buffer_size partition of calaman.feast.
 *
 * One caller-provided buffer. A single function, feast_detail::carve, both sizes
 * the layout (given a null base) and hands out the pointers (given the real
 * one), so the size query and the carving cannot drift apart. Every region
 * starts 256-byte aligned.
 *
 * The Ne resolvents dominate: Ne n^2 complex elements, 32 MiB at n = 512 in
 * double with Ne = 8. Everything else is O(n m0). The QR and eigensolver
 * workspaces are never live at the same time, so they share one region.
 *
 * STATUS TYPE. make_feast_slices / feast_bufferSize return a @ref calaman::Status,
 * like every other calaman.feast entry point: an argument fault surfaces as a BLAS
 * INVALID_VALUE / ALLOC_FAILED, and a non-success solver query (orthogonalize/syevd
 * bufferSize) flows through CLM_TRY in its own solver domain.
 */

module;

#include "feast_bridge.h"

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

#include <cstddef>

export module calaman.feast:buffer_size;

import std;
import wwr.blas;            // WWRBLAS_STATUS_*, wwrblasFillMode_t, WWRBLAS_FILL_MODE_*
import wwr.solver;          // wwrsolverDnHandle_t, wwrsolverEigMode_t, WWRSOLVER_EIG_MODE_VECTOR
import wwr.wrappers.common; // real_fp, RealToComplexType
import wwr.wrappers.solver; // syevd_bufferSize
import calaman.common;      // align_up
import calaman.orthogonalize; // orthogonalize_bufferSize
export import calaman.error_handling; // Status -- the cross-domain return type

export namespace calaman {

/**
 * @brief Pointers into the solver's single workspace buffer.
 *
 * Built once per solve by make_feast_slices and passed to each step, so the
 * steps allocate nothing and the sizing lives in one place.
 */
template<wwr::real_fp T>
struct FeastSlices {
  using C = wwr::RealToComplexType<T>;

  // ── the contour filter ─────────────────────────────────────────────────
  C *resolvents = nullptr;          ///< Ne packed n x n blocks: Z_e I - A, then its LU factors
  C *rhs = nullptr;                 ///< Ne n x m0 blocks: Y, then (Z_e I - A)^{-1} Y
  std::size_t resolvent_stride = 0; ///< elements from one resolvent block to the next
  std::size_t rhs_stride = 0;       ///< elements from one rhs block to the next
  C **resolvent_ptrs = nullptr;     ///< device array of the Ne resolvent addresses
  C **rhs_ptrs = nullptr;           ///< device array of the Ne rhs addresses
  int *ipiv = nullptr;              ///< Ne x n LU pivots, contiguous as getrfBatched wants

  // ── Rayleigh-Ritz ──────────────────────────────────────────────────────
  T *basis = nullptr;     ///< n x m0: the filtered subspace, orthonormalized in place
  T *a_basis = nullptr;   ///< n x m0: A * basis
  T *a_ritz = nullptr;    ///< n x m0: A * the Ritz vectors, for the residuals
  T *projected = nullptr; ///< m0 x m0: basis^T A basis, then its eigenvectors
  T *rotated = nullptr;   ///< m0 x m0: those eigenvectors, in-interval columns first
  T *ritz = nullptr;      ///< m0: the Ritz values, ascending, as syevd returns them
  T *residuals = nullptr; ///< m0: relative residual of each returned pair; 0 past status->m
  T *colsum = nullptr;    ///< n: per-column 1-norms of A
  T *norm_a = nullptr;    ///< one element: ||A||_1

  // ── the per-iteration status, and pointers into it ─────────────────────
  device::FeastStatus<T> *status = nullptr;
  int *lu_info = nullptr;  ///< status->lu_info
  int *qr_info = nullptr;  ///< status->qr_info
  int *eig_info = nullptr; ///< &status->eig_info

  // ── shared solver scratch ──────────────────────────────────────────────
  T *scratch = nullptr; ///< orthogonalize's workspace, then syevd's
  int lwork_qr = 0;     ///< orthogonalize's lwork, in elements of T
  int lwork_eig = 0;    ///< syevd's lwork, in elements of T
};

} // namespace calaman

namespace calaman::feast_detail {

/// Lay out the workspace from @p d_work, or from address zero to size it when
/// @p d_work is null. Returns the bytes the layout spans.
template<wwr::real_fp T>
std::size_t carve(void *d_work, const int n, const int m0, const int ne, const int lwork_qr,
                  const int lwork_eig, FeastSlices<T> *out) {
  using C = wwr::RealToComplexType<T>;
  constexpr std::size_t kAlign = 256;

  auto *const base = static_cast<std::byte *>(d_work);
  std::size_t offset = 0;
  const auto take = [&](const std::size_t bytes) -> std::byte * {
    std::byte *at = (base != nullptr) ? base + offset : nullptr;
    offset += align_up(bytes, kAlign);
    return at;
  };

  const std::size_t nz = static_cast<std::size_t>(n);
  const std::size_t m0z = static_cast<std::size_t>(m0);
  const std::size_t nez = static_cast<std::size_t>(ne);

  FeastSlices<T> s;
  s.lwork_qr = lwork_qr;
  s.lwork_eig = lwork_eig;

  // Every block starts on the alignment, which sizeof(C) divides, so the strides
  // are whole elements.
  s.resolvent_stride = align_up(nz * nz * sizeof(C), kAlign) / sizeof(C);
  s.rhs_stride = align_up(nz * m0z * sizeof(C), kAlign) / sizeof(C);
  s.resolvents = reinterpret_cast<C *>(take(s.resolvent_stride * sizeof(C) * nez));
  s.rhs = reinterpret_cast<C *>(take(s.rhs_stride * sizeof(C) * nez));
  s.resolvent_ptrs = reinterpret_cast<C **>(take(nez * sizeof(C *)));
  s.rhs_ptrs = reinterpret_cast<C **>(take(nez * sizeof(C *)));
  s.ipiv = reinterpret_cast<int *>(take(nez * nz * sizeof(int)));

  s.basis = reinterpret_cast<T *>(take(nz * m0z * sizeof(T)));
  s.a_basis = reinterpret_cast<T *>(take(nz * m0z * sizeof(T)));
  s.a_ritz = reinterpret_cast<T *>(take(nz * m0z * sizeof(T)));
  s.projected = reinterpret_cast<T *>(take(m0z * m0z * sizeof(T)));
  s.rotated = reinterpret_cast<T *>(take(m0z * m0z * sizeof(T)));
  s.ritz = reinterpret_cast<T *>(take(m0z * sizeof(T)));
  s.residuals = reinterpret_cast<T *>(take(m0z * sizeof(T)));
  s.colsum = reinterpret_cast<T *>(take(nz * sizeof(T)));
  s.norm_a = reinterpret_cast<T *>(take(sizeof(T)));

  std::byte *const status = take(sizeof(device::FeastStatus<T>));
  s.status = reinterpret_cast<device::FeastStatus<T> *>(status);
  if (status != nullptr) {
    s.lu_info = reinterpret_cast<int *>(status + offsetof(device::FeastStatus<T>, lu_info));
    s.qr_info = reinterpret_cast<int *>(status + offsetof(device::FeastStatus<T>, qr_info));
    s.eig_info = reinterpret_cast<int *>(status + offsetof(device::FeastStatus<T>, eig_info));
  }

  const int lwork = std::max(lwork_qr, lwork_eig);
  s.scratch = reinterpret_cast<T *>(take(static_cast<std::size_t>(lwork) * sizeof(T)));

  if (out != nullptr) {
    *out = s;
  }
  return offset;
}

} // namespace calaman::feast_detail

export namespace calaman {

/**
 * @brief Carve @p d_work into FeastSlices for an n x n problem with an m0-column
 *        subspace and Ne quadrature nodes.
 *
 * @param d_work      Workspace, or null to size it only.
 * @param slices      Out, may be null.
 * @param lwork_bytes Out, may be null: the bytes the layout needs.
 */
template<wwr::real_fp T, std::size_t Ne>
  requires(Ne == 4 || Ne == 8)
Status make_feast_slices(wwr::wwrsolverDnHandle_t cusolver_handle, const int n, const int m0,
                         void *d_work, FeastSlices<T> *slices, std::size_t *lwork_bytes) {
  if (n < 1 || m0 < 1 || m0 > n) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  int lwork_qr = 0;
  CLM_TRY(orthogonalize_bufferSize<T>(cusolver_handle, n, m0, &lwork_qr));

  int lwork_eig = 0;
  CLM_TRY(wwr::syevd_bufferSize<T>(cusolver_handle, wwr::WWRSOLVER_EIG_MODE_VECTOR,
                                   wwr::WWRBLAS_FILL_MODE_LOWER, m0, static_cast<T *>(nullptr), m0,
                                   static_cast<T *>(nullptr), &lwork_eig));

  const std::size_t bytes =
      feast_detail::carve<T>(d_work, n, m0, static_cast<int>(Ne), lwork_qr, lwork_eig, slices);
  if (lwork_bytes != nullptr) {
    *lwork_bytes = bytes;
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief Device workspace, in bytes, that feast_solver needs.
 *
 * @tparam T  Real floating-point type (float or double).
 * @tparam Ne Number of quadrature nodes (4 or 8); must match feast_solver's.
 * @param n           Matrix dimension.
 * @param m0          Subspace size, 1 <= m0 <= n.
 * @param lwork_bytes Out: bytes required.
 */
template<wwr::real_fp T, std::size_t Ne = 8>
  requires(Ne == 4 || Ne == 8)
Status feast_bufferSize(wwr::wwrsolverDnHandle_t cusolver_handle, const int n, const int m0,
                        std::size_t *lwork_bytes) {
  if (lwork_bytes == nullptr) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  return make_feast_slices<T, Ne>(cusolver_handle, n, m0, nullptr, nullptr, lwork_bytes);
}

} // namespace calaman
