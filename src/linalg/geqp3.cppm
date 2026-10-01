/**
 * @file geqp3.cppm
 * @brief The :geqp3 partition of calaman.linalg -- the top-level QR-with-column-
 *        pivoting driver, LAPACK's ?geqp3 (the all-free, blocked path)
 *
 * Factors an m-by-n column-major matrix with column pivoting into A*P = Q*R,
 * overwriting A the LAPACK way: R in the upper trapezoid, reflector j below the
 * diagonal of column j, the scalars in @p tau, the permutation in @p jpvt. Drives
 * the common nfxd == 0 case (@p jpvt all-free on entry): seeds the column norms
 * vn1/vn2 as the Euclidean norm of each column (one wwr::nrm2 each), then factors
 * the matrix in panels. Templated over float, double.
 *
 * BLOCKED (level-3). When min(m,n) exceeds kCrossoverBlockSize the driver runs a
 * loop of :laqps blocks of up to kBlockSize columns -- each block factors kb
 * columns (kb <= nb, possibly short if a column norm collapses) and applies its
 * effect to the rest of the trailing matrix with ONE gemm -- advancing the panel
 * offset by the returned kb, and finishes the tail (the last min(m,n) - offset
 * columns, below the crossover) with the unblocked :laqp2. When min(m,n) is at
 * or below the crossover the whole matrix is one :laqp2 panel, as before (#11).
 * This is the seam issue #12 fills: #11 shipped the unblocked driver with the
 * crossover held above every n; here the crossover is a real, small constant.
 *
 * A HOST COMPOSITION, not a kernel: the norm seed is a wwr::nrm2 loop, each panel
 * is :laqps or :laqp2, and the one host<->device sync per step is the panels'
 * own granularity. No .cu of its own (:laqps carries the downdate kernel).
 *
 * Allocation-free shipped surface (CLAUDE.md, test/shared/README.md): A, tau,
 * vn1, vn2 are caller-provided device pointers and jpvt a HOST int array. The
 * blocked path needs :laqps's extra scratch -- the auxiliary matrix F (n-by-nb),
 * the vector auxv (length nb) and the int degraded-column mask -- so the driver's
 * @p work contract is EXTENDED: work must hold, as one device T buffer,
 * n (laqp2's per-step larf intermediate) + n*nb (F) + nb (auxv) + the mask (n
 * ints, reinterpreted from n T slots, valid since sizeof(T) >= sizeof(int) for
 * float/double). The helper geqp3_work_size(m, n) returns that length; the test
 * allocates it. geqp3 seeds vn1/vn2 before handing the panels on; :laqp2
 * initialises jpvt, and :laqps only updates it, so for the blocked path geqp3
 * initialises jpvt to the identity itself before the first :laqps block.
 *
 * Requires the handle's DEFAULT (host) pointer mode, like larfg/laqp2/laqps, and
 * orders the per-column norm uploads on the handle's own stream. Complex ?geqp3
 * is a deliberate later extension, as larfg/laqp2 document.
 *
 * Usage:
 *   import calaman.linalg;
 *   import wwr.blas;          // wwrblasHandle_t, wwrblasCreate
 *   // d_A: m x n device matrix, lda; d_tau: length min(m,n);
 *   // d_vn1, d_vn2: length n (scratch, seeded here);
 *   // d_work: length calaman::geqp3_work_size(m, n); jpvt: host int[n]
 *   calaman::geqp3<double>(handle, m, n, d_A, lda, jpvt, d_tau, d_vn1, d_vn2,
 *                          d_work);
 */

export module calaman.linalg:geqp3;

import wwr.blas;          // wwrblasHandle_t, wwrblasStatus_t, WWRBLAS_STATUS_*
import wwr.runtime_api;   // wwrMemcpy(Async), wwrStreamSynchronize, wwrSuccess
import wwr.wrappers.blas; // nrm2
import :laqp2;            // calaman::laqp2
import :laqps;            // calaman::laqps
import std;              // std::min

namespace calaman {

/// @brief The block width the blocked path factors per :laqps call
///
/// Each :laqps block factors up to this many columns, deferring one gemm per
/// block. 32 is a conventional LAPACK-style panel width -- wide enough that the
/// level-3 gemm dominates the per-column level-2 work, narrow enough that F
/// (n-by-nb) stays small. The real LAPACK asks ilaenv; a fixed constant is the
/// honest first cut for a single-GPU library with no machine database.
inline constexpr int kBlockSize = 32;

/// @brief The crossover below which the matrix is one unblocked :laqp2 panel
///
/// Blocked QR only pays off once there is a trailing matrix large enough for the
/// deferred gemm to beat repeated level-2 updates; below this the bookkeeping
/// (F, the deferred gemm, the mask recompute) is pure overhead, so the driver
/// runs :laqp2 directly. Held deliberately SMALL (not the 1<<30 seam #11 used),
/// so any non-trivial matrix takes the blocked :laqps route and the oracle suite
/// exercises it. A caller that drives @p nb past min(m,n) (see geqp3's nb
/// parameter) forces the single-:laqp2 tail regardless, which the crossover test
/// uses to check the fallback still matches.
inline constexpr int kCrossoverBlockSize = 4;

/// @brief Length (in T elements) the @p work buffer geqp3 needs for an m-by-n A
///
/// Layout, as one contiguous device T buffer: [0, n) laqp2's per-step larf
/// intermediate; [n, n + n*nb) the auxiliary matrix F (n-by-nb, ldf = n);
/// [n + n*nb, n + n*nb + nb) auxv; [n + n*nb + nb, n + n*nb + nb + n) the int
/// degraded-column mask (n ints reinterpreted from n T slots). nb is kBlockSize
/// capped at min(m,n), matching the width geqp3 passes to :laqps.
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

/// @brief Factor A with QR column pivoting, all-free blocked (LAPACK ?geqp3)
///
/// Produces A*P = Q*R in place: R in A's upper trapezoid, reflector j below the
/// diagonal of column j, tau[j] the j-th scalar, jpvt the 1-based column
/// permutation. Seeds vn1[j] = vn2[j] = ||A(:,j)|| with one wwr::nrm2 per column,
/// then factors in :laqps blocks of @p nb columns (finishing the tail with
/// :laqp2) when min(m,n) > kCrossoverBlockSize, else as one :laqp2 panel.
///
/// Short-circuits: the first failing nrm2 / upload / panel status is returned and
/// the factorization stops there. Returns success and writes nothing when the
/// matrix is empty (m <= 0 or n <= 0). On a device-read/write failure that
/// surfaces no BLAS status returns WWRBLAS_STATUS_NOT_INITIALIZED.
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
///           min(m,n) forces the single-:laqp2 tail -- the crossover fallback
/// @return The status of the failing step, otherwise WWRBLAS_STATUS_SUCCESS
export template<typename T>
wwr::wwrblasStatus_t geqp3(wwr::wwrblasHandle_t handle, const int m, const int n, T *A,
                           const int lda, int *jpvt, T *tau, T *vn1, T *vn2, T *work,
                           const int nb = 0) {
  if (m <= 0 || n <= 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  wwr::wwrStream_t stream{};
  if (wwr::wwrblasGetStream(handle, &stream) != wwr::WWRBLAS_STATUS_SUCCESS) {
    return wwr::WWRBLAS_STATUS_NOT_INITIALIZED;
  }

  // Seed vn1[j] = vn2[j] = ||A(:,j)||_2, column by column: nrm2 writes the norm
  // to a host scalar (host pointer mode), then it is uploaded to both arrays.
  for (int j = 0; j < n; ++j) {
    T norm = T{0};
    const auto s = wwr::nrm2<T>(handle, m, A + static_cast<std::size_t>(j) * lda, 1, &norm);
    if (s != wwr::WWRBLAS_STATUS_SUCCESS) {
      return s;
    }
    if (wwr::wwrMemcpyAsync(vn1 + j, &norm, sizeof(T), wwr::wwrMemcpyHostToDevice, stream) !=
            wwr::wwrSuccess ||
        wwr::wwrMemcpyAsync(vn2 + j, &norm, sizeof(T), wwr::wwrMemcpyHostToDevice, stream) !=
            wwr::wwrSuccess ||
        wwr::wwrStreamSynchronize(stream) != wwr::wwrSuccess) {
      return wwr::WWRBLAS_STATUS_NOT_INITIALIZED;
    }
  }

  const int mn = std::min(m, n);
  const int block = nb > 0 ? nb : kBlockSize;

  // Below the crossover (or when the override widens nb past the whole panel),
  // the matrix is one unblocked :laqp2 panel: laqp2 initialises jpvt itself.
  if (mn <= kCrossoverBlockSize || block >= mn) {
    return laqp2<T>(handle, m, n, 0, A, lda, jpvt, tau, vn1, vn2, work);
  }

  // Blocked: :laqps only UPDATES jpvt (the driver owns the running permutation
  // across blocks), so seed the identity here before the first block.
  for (int j = 0; j < n; ++j) {
    jpvt[j] = j + 1;
  }

  // Carve the extended work buffer into laqp2 scratch | F | auxv | flags.
  const std::size_t nn = static_cast<std::size_t>(n);
  T *laqp2_work = work;
  T *F = laqp2_work + nn;
  const int ldf = n;
  T *auxv = F + nn * static_cast<std::size_t>(block);
  int *flags = reinterpret_cast<int *>(auxv + block);

  // Factor blocks of up to `block` columns until the panel is exhausted; a short
  // kb (a collapsed column) still advances by kb, and a kb of 0 cannot stall
  // (a nonempty panel always factors at least one column) -- guard anyway.
  int offset = 0;
  while (offset < mn) {
    const int want = std::min(block, mn - offset);
    int kb = 0;
    const auto s = laqps<T>(handle, m, n - offset, offset, want, &kb,
                            A + static_cast<std::size_t>(offset) * lda, lda, jpvt + offset,
                            tau + offset, vn1 + offset, vn2 + offset, F, ldf, auxv, flags);
    if (s != wwr::WWRBLAS_STATUS_SUCCESS) {
      return s;
    }
    if (kb <= 0) {
      break;
    }
    offset += kb;
    // A block that returned fewer columns than asked (a collapsing norm) means
    // the remaining panel is better finished unblocked -- fall through to the
    // :laqp2 tail below, exactly as LAPACK breaks its blocked loop.
    if (kb < want) {
      break;
    }
  }

  // Tail: the last n - offset columns, finished as one unblocked :laqp2 panel.
  // :laqp2 is written to run at offset 0 over a whole matrix -- it seeds its own
  // jpvt to the LOCAL identity and pivots all its columns -- so it is handed the
  // trailing SUBMATRIX view (A(1, offset+1), n - offset columns, offset 0), the
  // same slicing the :laqps blocks take. Its vn1/vn2 slices already describe
  // A(offset:m, :) from the blocks' downdates. That makes jpvt[offset:n] come
  // back as a LOCAL permutation of {1 .. n-offset}; remap it to the true
  // original indices via a snapshot of the pre-tail jpvt[offset:n].
  if (offset < mn) {
    const int tn = n - offset;
    std::vector<int> saved(static_cast<std::size_t>(tn));
    for (int p = 0; p < tn; ++p) {
      saved[static_cast<std::size_t>(p)] = jpvt[offset + p];
    }
    const auto s =
        laqp2<T>(handle, m, tn, 0, A + static_cast<std::size_t>(offset) * lda, lda, jpvt + offset,
                 tau + offset, vn1 + offset, vn2 + offset, work);
    if (s != wwr::WWRBLAS_STATUS_SUCCESS) {
      return s;
    }
    // jpvt[offset + p] is now a 1-based LOCAL index into the pre-tail columns
    // offset..n-1; translate it back to the true original column index.
    for (int p = 0; p < tn; ++p) {
      const int loc = jpvt[offset + p]; // 1..tn
      jpvt[offset + p] = saved[static_cast<std::size_t>(loc - 1)];
    }
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
