/**
 * @file geqp3.cppm
 * @brief The :geqp3 partition of calaman.linalg -- the top-level QR-with-column-
 *        pivoting driver, LAPACK's ?geqp3 (the all-free, unblocked path)
 *
 * Factors an m-by-n column-major matrix with column pivoting into A*P = Q*R,
 * overwriting A the LAPACK way: R in the upper trapezoid, reflector j below the
 * diagonal of column j, the scalars in @p tau, the permutation in @p jpvt. Drives
 * the common nfxd == 0 case (@p jpvt all-free on entry): seeds the column norms
 * vn1/vn2 as the Euclidean norm of each column (one wwr::nrm2 each), then runs the
 * merged :laqp2 panel over all n columns at offset 0. Templated over float, double.
 *
 * A HOST COMPOSITION, not a kernel: the norm seed is a wwr::nrm2 loop, the factor
 * is :laqp2, and the one host<->device sync per column is :laqp2's own granularity.
 * No .cu of its own.
 *
 * UNBLOCKED FOR NOW: kCrossoverBlockSize is a tuning constant held above any n
 * this driver sees, so the blocked ?laqps crossover LAPACK reaches via ilaenv
 * never fires. It is the seam issue #12 replaces with the real block-size logic;
 * #13 adds the fixed-prefix (nfxd > 0) handling this path omits.
 *
 * Allocation-free shipped surface (CLAUDE.md, test/shared/README.md): A, tau,
 * vn1, vn2, work are caller-provided device pointers and jpvt a HOST int array,
 * exactly :laqp2's contract -- geqp3 adds no scratch, it only fills vn1/vn2 before
 * handing them on. :laqp2 initialises jpvt and fills it with the permutation
 * LAPACKE_?geqp3 returns for an all-free entry, so the oracle agrees.
 *
 * Requires the handle's DEFAULT (host) pointer mode, like larfg/laqp2, and orders
 * the per-column norm uploads on the handle's own stream. Complex ?geqp3 is a
 * deliberate later extension, as larfg/laqp2 document.
 *
 * Usage:
 *   import calaman.linalg;
 *   import wwr.blas;          // wwrblasHandle_t, wwrblasCreate
 *   // d_A: m x n device matrix, lda; d_tau: length min(m,n);
 *   // d_vn1, d_vn2: length n (scratch, seeded here); d_work: length n;
 *   // jpvt: host int[n]
 *   calaman::geqp3<double>(handle, m, n, d_A, lda, jpvt, d_tau, d_vn1, d_vn2,
 *                          d_work);
 */

export module calaman.linalg:geqp3;

import wwr.blas;          // wwrblasHandle_t, wwrblasStatus_t, WWRBLAS_STATUS_*
import wwr.runtime_api;   // wwrMemcpy(Async), wwrStreamSynchronize, wwrSuccess
import wwr.wrappers.blas; // nrm2
import :laqp2;            // calaman::laqp2
import std;              // std::min

namespace calaman {

/// @brief The block-size crossover this driver tunes to -- unblocked for now
///
/// LAPACK's ?geqp3 asks ilaenv for a panel block size and, past it, runs the
/// blocked ?laqps; issue #12 wires that. Held above any n reached here, so the
/// crossover never fires and every matrix takes the unblocked :laqp2 route. This
/// is the seam #12 replaces with the real block-size logic.
inline constexpr int kCrossoverBlockSize = 1 << 30;

/// @brief Factor A with QR column pivoting, all-free unblocked (LAPACK ?geqp3)
///
/// Produces A*P = Q*R in place: R in A's upper trapezoid, reflector j below the
/// diagonal of column j, tau[j] the j-th scalar, jpvt the 1-based column
/// permutation. Seeds vn1[j] = vn2[j] = ||A(:,j)|| with one wwr::nrm2 per column,
/// then runs :laqp2 over all n columns at offset 0. The @p jpvt all-free case
/// (nfxd == 0): :laqp2 initialises jpvt itself.
///
/// Short-circuits: the first failing nrm2 / upload / laqp2 status is returned and
/// the factorization stops there. Returns success and writes nothing when the
/// matrix is empty (m <= 0 or n <= 0). On a device-read/write failure that
/// surfaces no BLAS status returns WWRBLAS_STATUS_NOT_INITIALIZED, the neutral
/// code larfg/laqp2/diff_norm use.
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
/// @param work Device workspace, length >= n; laqp2's per-step intermediate
/// @return The status of the failing step, otherwise WWRBLAS_STATUS_SUCCESS
export template<typename T>
wwr::wwrblasStatus_t geqp3(wwr::wwrblasHandle_t handle, const int m, const int n, T *A,
                           const int lda, int *jpvt, T *tau, T *vn1, T *vn2, T *work) {
  if (m <= 0 || n <= 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  // Order the per-column norm uploads on the handle's own stream, so they follow
  // the caller's uploads and this routine's nrm2 work -- the larfg discipline.
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

  // Unblocked: the whole matrix is one panel. kCrossoverBlockSize sits above n,
  // so there is no blocked :laqps leg yet (issue #12). Factor all n columns from
  // offset 0; laqp2 fills jpvt, R, tau and updates vn1/vn2 as it pivots.
  (void)kCrossoverBlockSize;
  return laqp2<T>(handle, m, n, 0, A, lda, jpvt, tau, vn1, vn2, work);
}

} // namespace calaman
