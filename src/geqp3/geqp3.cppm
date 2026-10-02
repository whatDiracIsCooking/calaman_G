/**
 * @file geqp3.cppm
 * @brief The calaman.geqp3 module -- the top-level QR-with-column-pivoting
 *        driver, LAPACK's ?geqp3 (the all-free path and the
 *        caller-fixed-prefix path)
 *
 * Factors an m-by-n column-major matrix with column pivoting into A*P = Q*R,
 * overwriting A the LAPACK way: R in the upper trapezoid, reflector j below the
 * diagonal of column j, the scalars in @p tau, the permutation in @p jpvt.
 *
 * Two regimes, split by @p jpvt on entry (LAPACK's ?geqp3 convention):
 *
 *   nfxd == 0 (ALL FREE): the 11-arg overload. Every @p jpvt entry is treated as
 *   free; it seeds the column norms vn1/vn2 as the Euclidean norm of each column
 *   (one wwr::nrm2 each) and factors the matrix in panels (below). This is the
 *   #11/#12 driver, unchanged.
 *
 *   nfxd > 0 (FIXED PREFIX): the 15-arg overload, which additionally takes a
 *   wwr::wwrsolverDnHandle_t and a solver workspace. On entry jpvt[j] != 0 marks
 *   column j as a caller-FIXED leading column (it must land in the leading
 *   positions, in the given order); jpvt[j] == 0 marks a free column. The
 *   factorization is pure REUSE of already-wrapped solver routines plus host
 *   column bookkeeping -- no new numerical kernel:
 *     1. Pre-permute: move every fixed column to the front with a sequence of
 *        column wwr::swap's, maintaining jpvt so the final jpvt still reports the
 *        true original index of each column. nfxd = number of fixed columns.
 *     2. Factor the fixed block A(:, 0:nfxd) with the wrapped wwr::geqrf -- an
 *        ordinary unpivoted QR of the leading block.
 *     3. Apply Q^T to the trailing free columns: A(:, nfxd:n) := Q^T * A(:, nfxd:n)
 *        via wwr::ormqr (side = Left, trans = transpose). wwr::unmqr is the complex
 *        counterpart; templated over float/double, ormqr suffices.
 *     4. Hand the trailing free submatrix A(nfxd:m, nfxd:n) to the EXISTING pivoted
 *        path at offset = nfxd, with the free-column vn1/vn2 seeded from the
 *        trailing part A(nfxd:m, free col). The all-free blocked driver already
 *        factors a trailing submatrix at an offset, so this reuses it directly.
 *   The 15-arg overload with an all-free jpvt (nfxd == 0) just forwards to the
 *   11-arg body, so a caller that always has the solver handle can use it alone.
 *
 * Templated over float, double.
 *
 * BLOCKED (level-3). When min(m,n) exceeds kCrossoverBlockSize the driver runs a
 * loop of laqps blocks of up to kBlockSize columns -- each block factors kb
 * columns (kb <= nb, possibly short if a column norm collapses) and applies its
 * effect to the rest of the trailing matrix with ONE gemm -- advancing the panel
 * offset by the returned kb, and finishes the tail (the last min(m,n) - offset
 * columns, below the crossover) with the unblocked laqp2. When min(m,n) is at
 * or below the crossover the whole matrix is one laqp2 panel, as before (#11).
 * This is the seam issue #12 fills: #11 shipped the unblocked driver with the
 * crossover held above every n; here the crossover is a real, small constant.
 *
 * A HOST COMPOSITION, not a kernel: the norm seed is a wwr::nrm2 loop, each panel
 * is laqps or laqp2, and the one host<->device sync per step is the panels'
 * own granularity. No .cu of its own (laqps carries the downdate kernel).
 *
 * Allocation-free shipped surface (CLAUDE.md, test/shared/README.md): A, tau,
 * vn1, vn2 are caller-provided device pointers and jpvt a HOST int array. The
 * blocked path needs laqps's extra scratch -- the auxiliary matrix F (n-by-nb),
 * the vector auxv (length nb) and the int degraded-column mask -- so the driver's
 * @p work contract is EXTENDED: work must hold, as one device T buffer,
 * n (laqp2's per-step larf intermediate) + n*nb (F) + nb (auxv) + the mask (n
 * ints, reinterpreted from n T slots, valid since sizeof(T) >= sizeof(int) for
 * float/double). The helper geqp3_work_size(m, n) returns that length; the test
 * allocates it. geqp3 seeds vn1/vn2 before handing the panels on; laqp2
 * initialises jpvt, and laqps only updates it, so for the blocked path geqp3
 * initialises jpvt to the identity itself before the first laqps block.
 *
 * Requires the handle's DEFAULT (host) pointer mode, like larfg/laqp2/laqps, and
 * orders the per-column norm uploads on the handle's own stream. Complex ?geqp3
 * is a deliberate later extension, as larfg/laqp2 document.
 *
 * Usage (all-free, 11-arg):
 *   import calaman.geqp3;
 *   import wwr.blas;          // wwrblasHandle_t, wwrblasCreate
 *   // d_A: m x n device matrix, lda; d_tau: length min(m,n);
 *   // d_vn1, d_vn2: length n (scratch, seeded here);
 *   // d_work: length calaman::geqp3_work_size(m, n); jpvt: host int[n]
 *   calaman::geqp3<double>(handle, m, n, d_A, lda, jpvt, d_tau, d_vn1, d_vn2,
 *                          d_work);
 *
 * Usage (fixed prefix, 15-arg):
 *   import wwr.solver;        // wwrsolverDnHandle_t, wwrsolverDnCreate/SetStream
 *   import wwr.wrappers.blas; // wwrblasSideMode_t, wwrblasOperation_t
 *   // solver: a wwrsolverDnHandle_t bound to the SAME stream as `handle`;
 *   // jpvt[j] != 0 marks a fixed leading column, jpvt[j] == 0 a free one;
 *   // d_swork: device solver workspace, length >= calaman::geqp3_solver_work_size
 *   //          (solver, m, n); d_info: device int (the geqrf/ormqr devInfo)
 *   calaman::geqp3<double>(handle, solver, m, n, d_A, lda, jpvt, d_tau, d_vn1,
 *                          d_vn2, d_work, d_swork, swork_len, d_info);
 */

module;

#include "error_handling/error_macros.h" // CLM_TRY -- a macro, arrives by #include, not import

export module calaman.geqp3;

import wwr.blas;            // wwrblasHandle_t, wwrblasStatus_t, WWRBLAS_STATUS_*
import wwr.solver;          // wwrsolverDnHandle_t, wwrsolverStatus_t, WWRSOLVER_*
import wwr.runtime_api;     // wwrMemcpy(Async), wwrStreamSynchronize, wwrSuccess
import wwr.wrappers.blas;   // nrm2, wwrblasSideMode_t, wwrblasOperation_t
import wwr.wrappers.solver; // geqrf, geqrf_bufferSize, ormqr, ormqr_bufferSize
import calaman.laqp2;       // calaman::laqp2
import calaman.laqps;       // calaman::laqps
import std;                // std::min

// export import, not plain: the geqp3 drivers RETURN calaman::Status, whose
// member functions a consumer must see, and it supplies the ::calaman::Status
// CLM_TRY names. (calaman.laqps already re-exports it, but geqp3 names the type
// directly, so it declares the dependency itself.)
export import calaman.error_handling; // calaman::Status -- the return type

namespace calaman {

/// @brief The block width the blocked path factors per laqps call
///
/// Each laqps block factors up to this many columns, deferring one gemm per
/// block. 32 is a conventional LAPACK-style panel width -- wide enough that the
/// level-3 gemm dominates the per-column level-2 work, narrow enough that F
/// (n-by-nb) stays small. The real LAPACK asks ilaenv; a fixed constant is the
/// honest first cut for a single-GPU library with no machine database.
inline constexpr int kBlockSize = 32;

/// @brief The crossover below which the matrix is one unblocked laqp2 panel
///
/// Blocked QR only pays off once there is a trailing matrix large enough for the
/// deferred gemm to beat repeated level-2 updates; below this the bookkeeping
/// (F, the deferred gemm, the mask recompute) is pure overhead, so the driver
/// runs laqp2 directly. Held deliberately SMALL (not the 1<<30 seam #11 used),
/// so any non-trivial matrix takes the blocked laqps route and the oracle suite
/// exercises it. A caller that drives @p nb past min(m,n) (see geqp3's nb
/// parameter) forces the single-laqp2 tail regardless, which the crossover test
/// uses to check the fallback still matches.
inline constexpr int kCrossoverBlockSize = 4;

/// @brief Length (in T elements) the @p work buffer geqp3 needs for an m-by-n A
///
/// Layout, as one contiguous device T buffer: [0, n) laqp2's per-step larf
/// intermediate; [n, n + n*nb) the auxiliary matrix F (n-by-nb, ldf = n);
/// [n + n*nb, n + n*nb + nb) auxv; [n + n*nb + nb, n + n*nb + nb + n) the int
/// degraded-column mask (n ints reinterpreted from n T slots). nb is kBlockSize
/// capped at min(m,n), matching the width geqp3 passes to laqps.
export inline constexpr std::size_t geqp3_work_size(const int m, const int n) {
  if (m <= 0 || n <= 0) {
    return 1;
  }
  const int nb = std::min(kBlockSize, std::min(m, n));
  const std::size_t nn = static_cast<std::size_t>(n);
  return nn                                              // larf / laqp2 scratch
         + nn * static_cast<std::size_t>(nb)             // F
         + static_cast<std::size_t>(nb)                  // auxv
         + nn;                                           // flags (ints in T slots)
}

/// @brief Upper bound (in T elements) on the solver workspace the fixed-prefix
///        path needs for an m-by-n A, querying geqrf and ormqr at their maxima
///
/// The fixed-prefix overload runs one wwr::geqrf on the leading block (at most
/// min(m, n) columns) and one wwr::ormqr applying Q^T to the free tail; each
/// needs a workspace sized by its *_bufferSize query. The worst case is the
/// whole matrix fixed (geqrf of A(:, 0:min(m,n))) and the whole trailing matrix
/// free (ormqr over an m-by-n C with k = min(m,n) reflectors), so this queries
/// both at those maxima and returns the larger. It runs on the live solver
/// handle, so it returns 1 (a harmless floor) on any query failure or empty A --
/// the overload re-checks the status when it queries for the real sizes.
///
/// @param solver GPU solver handle; the *_bufferSize queries run on it
/// @param m Row count of A
/// @param n Column count of A
export template<typename T>
int geqp3_solver_work_size(wwr::wwrsolverDnHandle_t solver, const int m, const int n) {
  if (m <= 0 || n <= 0) {
    return 1;
  }
  const int mn = std::min(m, n);
  int geqrf_lwork = 0;
  int ormqr_lwork = 0;
  if (wwr::geqrf_bufferSize<T>(solver, m, mn, nullptr, m, &geqrf_lwork) !=
      wwr::WWRSOLVER_STATUS_SUCCESS) {
    return 1;
  }
  if (wwr::ormqr_bufferSize<T>(solver, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_OP_T, m, n, mn, nullptr,
                               m, nullptr, nullptr, m, &ormqr_lwork) !=
      wwr::WWRSOLVER_STATUS_SUCCESS) {
    return 1;
  }
  return std::max(std::max(geqrf_lwork, ormqr_lwork), 1);
}

/// @brief Factor A with QR column pivoting, all-free blocked (LAPACK ?geqp3)
///
/// Produces A*P = Q*R in place: R in A's upper trapezoid, reflector j below the
/// diagonal of column j, tau[j] the j-th scalar, jpvt the 1-based column
/// permutation. Seeds vn1[j] = vn2[j] = ||A(:,j)|| with one wwr::nrm2 per column,
/// then factors in laqps blocks of @p nb columns (finishing the tail with
/// laqp2) when min(m,n) > kCrossoverBlockSize, else as one laqp2 panel.
///
/// Short-circuits: the first failing nrm2 / upload / panel status is returned and
/// the factorization stops there. Returns success and writes nothing when the
/// matrix is empty (m <= 0 or n <= 0). A device-read/write failure surfaces its
/// own runtime error through the Status return.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle GPU BLAS handle in host pointer mode; A, tau, vn1, vn2, work live on its device
/// @param m Row count of A
/// @param n Column count of A
/// @param A Device matrix, m by n, column-major, overwritten with R and the reflectors
/// @param lda Leading dimension of A (>= m)
/// @param jpvt Host int array, length n; filled with the 1-based column permutation
/// @param tau Device array, length >= min(m, n); the reflector scalars
/// @param vn1 Device array, length n; partial column norms, seeded here
/// @param vn2 Device array, length n; original column norms, seeded here
/// @param work Device workspace, length >= geqp3_work_size(m, n)
/// @param nb Block width override; 0 (the default) uses kBlockSize. A value past
///           min(m,n) forces the single-laqp2 tail -- the crossover fallback
/// @return A calaman::Status: the failing step's status (in its own error
///         domain), otherwise success
namespace detail {

/// @brief Factor the free panel A(start:m, start:n) with Businger-Golub pivoting
///
/// The shared engine behind both geqp3 overloads: it pivots and factors the
/// columns [start, n) over the FULL m rows (pivot rows begin at row @p start, so
/// column swaps are full-length and rows above @p start -- already-factored R --
/// are preserved), reusing laqps blocks of @p block columns finishing with a
/// laqp2 tail above the crossover, else one laqp2 panel. The free columns'
/// vn1/vn2 (indices [start, n)) must be pre-seeded to describe A(start:m, col).
/// jpvt[start:n] is OVERWRITTEN with a LOCAL 1-based permutation of {1..n-start}
/// (position start+p holds which pre-call free column, 1-based within the panel,
/// now sits there); the caller remaps it to true original indices. For the
/// all-free overload start == 0 and that local permutation IS the answer.
template<typename T>
calaman::Status factor_free_panel(wwr::wwrblasHandle_t handle, const int m, const int n,
                                  const int start, T *A, const int lda, int *jpvt, T *tau, T *vn1,
                                  T *vn2, T *work, const int block) {
  const int mn = std::min(m - start, n - start); // pivot steps over the free panel
  if (mn <= 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  // Below the crossover (or an nb override past the panel), one unblocked laqp2
  // panel at offset=start over the full-m columns: laqp2 seeds jpvt[start:] to
  // the LOCAL identity {1..n-start} and pivots full-m columns itself.
  if (mn <= kCrossoverBlockSize || block >= mn) {
    return laqp2<T>(handle, m, n - start, start, A + static_cast<std::size_t>(start) * lda, lda,
                    jpvt + start, tau + start, vn1 + start, vn2 + start, work);
  }

  // Blocked: laqps only UPDATES jpvt (the driver owns the running permutation
  // across blocks), so seed the LOCAL identity over [start, n) before block one.
  for (int j = start; j < n; ++j) {
    jpvt[j] = j - start + 1;
  }

  // Carve the extended work buffer into laqp2 scratch | F | auxv | flags. F is
  // n-by-block (ldf = n covers the widest trailing slice the blocked loop uses).
  const std::size_t nn = static_cast<std::size_t>(n);
  T *laqp2_work = work;
  T *F = laqp2_work + nn;
  const int ldf = n;
  T *auxv = F + nn * static_cast<std::size_t>(block);
  int *flags = reinterpret_cast<int *>(auxv + block);

  // Factor blocks of up to `block` columns until the panel is exhausted; a short
  // kb (a collapsed column) still advances by kb, and a kb of 0 cannot stall
  // (a nonempty panel always factors at least one column) -- guard anyway.
  int offset = start;
  while (offset < start + mn) {
    const int want = std::min(block, start + mn - offset);
    int kb = 0;
    CLM_TRY(laqps<T>(handle, m, n - offset, offset, want, &kb,
                     A + static_cast<std::size_t>(offset) * lda, lda, jpvt + offset, tau + offset,
                     vn1 + offset, vn2 + offset, F, ldf, auxv, flags));
    if (kb <= 0) {
      break;
    }
    offset += kb;
    // A block that returned fewer columns than asked (a collapsing norm) means
    // the remaining panel is better finished unblocked -- fall through to the
    // laqp2 tail below, exactly as LAPACK breaks its blocked loop.
    if (kb < want) {
      break;
    }
  }

  // Tail: the remaining columns [offset, n), finished as one laqp2 panel at
  // offset=offset over the full-m columns, so its full-length swaps preserve the
  // R rows above `offset` that the blocks already filled. laqp2 reseeds
  // jpvt[offset:] to the LOCAL identity {1 .. n-offset}, so it comes back as a
  // permutation of those local indices; translate it through a snapshot of the
  // pre-tail jpvt[offset:n] (themselves panel-local {1..n-start} indices) so the
  // whole jpvt[start:n] stays a single consistent panel-local permutation.
  if (offset < start + mn) {
    const int tn = n - offset;
    std::vector<int> saved(static_cast<std::size_t>(tn));
    for (int p = 0; p < tn; ++p) {
      saved[static_cast<std::size_t>(p)] = jpvt[offset + p];
    }
    CLM_TRY(laqp2<T>(handle, m, tn, offset, A + static_cast<std::size_t>(offset) * lda, lda,
                     jpvt + offset, tau + offset, vn1 + offset, vn2 + offset, work));
    // jpvt[offset + p] is now a 1-based LOCAL index into the pre-tail columns
    // offset..n-1; translate it back to the panel-local index via `saved`.
    for (int p = 0; p < tn; ++p) {
      const int loc = jpvt[offset + p]; // 1..tn
      jpvt[offset + p] = saved[static_cast<std::size_t>(loc - 1)];
    }
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief Seed vn1[j] = vn2[j] = ||A(start:m, j)||_2 for the free columns [c0, n)
///
/// One wwr::nrm2 per column (host pointer mode writes a host scalar), uploaded to
/// both partial-norm arrays on @p stream. @p start is the first row the norm
/// counts (0 for the all-free seed, nfxd for the free tail after ormqr).
template<typename T>
calaman::Status seed_free_norms(wwr::wwrblasHandle_t handle, wwr::wwrStream_t stream, const int m,
                                const int n, const int start, const int c0, T *A, const int lda,
                                T *vn1, T *vn2) {
  const int rows = m - start;
  for (int j = c0; j < n; ++j) {
    T norm = T{0};
    if (rows > 0) {
      CLM_TRY(wwr::nrm2<T>(handle, rows, A + static_cast<std::size_t>(j) * lda + start, 1, &norm));
    }
    CLM_TRY(wwr::wwrMemcpyAsync(vn1 + j, &norm, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
    CLM_TRY(wwr::wwrMemcpyAsync(vn2 + j, &norm, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace detail

export template<typename T>
calaman::Status geqp3(wwr::wwrblasHandle_t handle, const int m, const int n, T *A, const int lda,
                      int *jpvt, T *tau, T *vn1, T *vn2, T *work, const int nb = 0) {
  if (m <= 0 || n <= 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(handle, &stream));

  // Seed vn1[j] = vn2[j] = ||A(:,j)||_2 for every (free) column, then factor the
  // whole panel at offset 0. With start == 0 the panel-local permutation
  // factor_free_panel returns IS the final jpvt, so no remap is needed.
  CLM_TRY(detail::seed_free_norms<T>(handle, stream, m, n, 0, 0, A, lda, vn1, vn2));
  const int block = nb > 0 ? nb : kBlockSize;
  return detail::factor_free_panel<T>(handle, m, n, 0, A, lda, jpvt, tau, vn1, vn2, work, block);
}

/// @brief Factor A with QR column pivoting, caller-FIXED leading columns
///        (LAPACK ?geqp3, the nfxd > 0 regime)
///
/// Produces A*P = Q*R in place with the columns the caller marked fixed landing
/// in the leading positions, in their given order, and the rest pivoted. On
/// entry jpvt[j] != 0 marks column j as a fixed leading column, jpvt[j] == 0 a
/// free one (LAPACK's ?geqp3 convention, which LAPACKE_?geqp3 honours, so it is a
/// faithful oracle for this regime). On exit jpvt is the 1-based permutation:
/// jpvt[k] is the true original index of the column now in position k.
///
/// Pure reuse (no new kernel): pre-permute the fixed columns to the front with a
/// sequence of column wwr::swap's, factor the fixed block A(:, 0:nfxd) with
/// wwr::geqrf, apply Q^T to the free tail A(:, nfxd:n) with wwr::ormqr, then hand
/// the trailing free submatrix A(nfxd:m, nfxd:n) to the all-free overload above
/// (seeding its vn1/vn2 from that submatrix's columns), remapping its local pivot
/// permutation back to the true original indices. With nfxd == 0 it forwards
/// straight to the all-free overload.
///
/// Short-circuits on the first failing step. Returns success and writes nothing
/// for an empty matrix. A geqrf/ormqr status that is not success surfaces through
/// the Status return in its OWN (solver) domain, no longer flattened to a BLAS
/// stand-in. @p info is the geqrf/ormqr devInfo scratch (0 on a well-formed QR,
/// which an unpivoted leading block always is).
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle GPU BLAS handle in host pointer mode; the device arrays live on its stream
/// @param solver GPU solver handle bound to the SAME stream as @p handle
/// @param m Row count of A
/// @param n Column count of A
/// @param A Device matrix, m by n, column-major, overwritten with R and the reflectors
/// @param lda Leading dimension of A (>= m)
/// @param jpvt Host int array, length n; in: nonzero marks a fixed column; out: the permutation
/// @param tau Device array, length >= min(m, n); the reflector scalars
/// @param vn1 Device array, length n; partial column norms (scratch, seeded here)
/// @param vn2 Device array, length n; original column norms (scratch, seeded here)
/// @param work Device workspace, length >= geqp3_work_size(m, n)
/// @param swork Device solver workspace, length >= geqp3_solver_work_size<T>(solver, m, n)
/// @param lwork_solver Length of @p swork in T elements
/// @param info Device int; the geqrf/ormqr devInfo (0 on success)
/// @param nb Block width override forwarded to the free-tail factorization
/// @return A calaman::Status: the failing step's status (in its own error
///         domain), otherwise success
export template<typename T>
calaman::Status geqp3(wwr::wwrblasHandle_t handle, wwr::wwrsolverDnHandle_t solver, const int m,
                      const int n, T *A, const int lda, int *jpvt, T *tau, T *vn1, T *vn2, T *work,
                      T *swork, const int lwork_solver, int *info, const int nb = 0) {
  if (m <= 0 || n <= 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(handle, &stream));

  // origin[k] tracks the true original index (1-based) of the column currently in
  // position k, so after the pre-permute and the free-tail factorization jpvt can
  // report true indices. It starts as the identity; `fixed` captures the caller's
  // entry marks BEFORE jpvt is used as output, since jpvt is both in and out.
  std::vector<int> origin(static_cast<std::size_t>(n));
  std::vector<bool> fixed(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    origin[static_cast<std::size_t>(j)] = j + 1;
    fixed[static_cast<std::size_t>(j)] = jpvt[j] != 0;
  }

  // Pre-permute: slide every fixed column to the front with a sequence of
  // full-column swaps, carrying `origin` and `fixed` alongside so the leading
  // block ends up in the caller's given order (the scan is left to right, so
  // fixed columns keep their relative order) and the true-origin bookkeeping
  // stays consistent. nfxd counts the fixed columns.
  int nfxd = 0;
  for (int j = 0; j < n; ++j) {
    if (!fixed[static_cast<std::size_t>(j)]) {
      continue;
    }
    if (j != nfxd) {
      CLM_TRY(wwr::swap<T, int>(handle, m, A + static_cast<std::size_t>(nfxd) * lda, 1,
                                A + static_cast<std::size_t>(j) * lda, 1));
      std::swap(origin[static_cast<std::size_t>(nfxd)], origin[static_cast<std::size_t>(j)]);
      // Positions [0,nfxd) are already-placed fixed columns, so position nfxd
      // holds a FREE column; after the swap the fixed one is at nfxd and the free
      // one moves to j. Track the mark so a later j still sees the truth there.
      fixed[static_cast<std::size_t>(j)] = fixed[static_cast<std::size_t>(nfxd)];
      fixed[static_cast<std::size_t>(nfxd)] = true;
    }
    ++nfxd;
  }

  // No fixed columns: nothing to pre-factor, forward to the all-free overload
  // (which re-seeds jpvt to the identity and seeds the norms itself).
  if (nfxd == 0) {
    return geqp3<T>(handle, m, n, A, lda, jpvt, tau, vn1, vn2, work, nb);
  }

  // Factor the fixed leading block A(:, 0:nfxd) with an ordinary unpivoted QR.
  // geqrf writes tau[0:nfxd] and the reflectors below the diagonal; its workspace
  // is the caller's solver scratch, sized by geqp3_solver_work_size.
  CLM_TRY(wwr::geqrf<T>(solver, m, nfxd, A, lda, tau, swork, lwork_solver, info));

  // Apply Q^T from the fixed block to the free tail: A(:, nfxd:n) := Q^T * A(:,
  // nfxd:n) (side = Left, trans = transpose, k = nfxd reflectors). unmqr is the
  // complex counterpart; float/double take ormqr.
  const int nfree = n - nfxd;
  if (nfree > 0) {
    CLM_TRY(wwr::ormqr<T>(solver, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_OP_T, m, nfree, nfxd, A, lda,
                          tau, A + static_cast<std::size_t>(nfxd) * lda, lda, swork, lwork_solver,
                          info));

    // Seed the free columns' partial norms from the TRAILING part A(nfxd:m, j)
    // (the ormqr output below the fixed block's R), then pivot-factor the free
    // panel A(nfxd:m, nfxd:n) at offset = nfxd over the FULL-m columns. Running at
    // offset nfxd keeps the swaps full-length, so the R rows 0:nfxd of the free
    // columns move with their column -- exactly LAPACK's ?laqp2/?laqps at
    // offset = nfxd. jpvt[nfxd:n] comes back as a LOCAL permutation of {1..nfree}.
    CLM_TRY(detail::seed_free_norms<T>(handle, stream, m, n, nfxd, nfxd, A, lda, vn1, vn2));
    const int block = nb > 0 ? nb : kBlockSize;
    std::vector<int> local(static_cast<std::size_t>(n), 0);
    CLM_TRY(detail::factor_free_panel<T>(handle, m, n, nfxd, A, lda, local.data(), tau, vn1, vn2,
                                         work, block));
    // Remap the free panel's local permutation to true original indices.
    // local[nfxd + p] (1..nfree) selects the (nfxd + local[..] - 1)-th column of
    // the pre-tail layout, whose true origin is origin[nfxd + local[..] - 1].
    std::vector<int> free_origin(static_cast<std::size_t>(nfree));
    for (int p = 0; p < nfree; ++p) {
      free_origin[static_cast<std::size_t>(p)] = origin[static_cast<std::size_t>(nfxd + p)];
    }
    for (int p = 0; p < nfree; ++p) {
      const int loc = local[static_cast<std::size_t>(nfxd + p)]; // 1..nfree
      origin[static_cast<std::size_t>(nfxd + p)] = free_origin[static_cast<std::size_t>(loc - 1)];
    }
  }

  // Publish the true-origin permutation: the leading nfxd positions keep their
  // (unpivoted) origins, the free tail carries the remapped ones.
  for (int j = 0; j < n; ++j) {
    jpvt[j] = origin[static_cast<std::size_t>(j)];
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
