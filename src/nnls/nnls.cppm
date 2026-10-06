/**
 * @file nnls.cppm
 * @brief The calaman.nnls module -- non-negative least squares:
 *        min ||A x - b||_2 subject to x >= 0, by the Lawson-Hanson active set
 *
 * Lawson & Hanson, *Solving Least Squares Problems* (1974), ch. 23 -- the
 * algorithm `scipy.optimize.nnls` uses. Real only (`float`/`double`): the x >= 0
 * constraint has no complex meaning. A is dense, column-major, any shape, and
 * may be rank-deficient.
 *
 * Let P be the passive set (free variables), Z the zero set. From x = 0,
 * P = empty, each outer iteration: (1) w = A^T(b - Ax); (2) stop if Z is empty
 * or max_{j in Z} w_j <= tol; (3) move argmax_{j in Z} w_j from Z to P; (4)
 * solve the unconstrained least-squares subproblem on the columns in P, and if
 * any component turns negative take the largest feasible step toward it, move
 * the zeroed entries back to Z, and re-solve.
 *
 * The P-subproblem reuses calaman::geqp3 (pivoted QR) rather than a plain QR, so
 * a rank-deficient passive set is handled by the same path: the rank is read off
 * the pivoted R diagonal and the deficient tail of the back-substitution is
 * truncated to zero (the minimum-norm choice). wwr::ormqr applies Q^T to b,
 * wwr::trsv back-substitutes, and the P/Z bookkeeping (gather, masked argmax,
 * min-ratio test, compaction) is the device library in nnls.cu.
 *
 * Host-synchronized, like geqp3: the loop branches on device data and the
 * subproblem size changes every step, so a few small scalars are read back each
 * iteration. Requires the blas handle's DEFAULT (host) pointer mode, and a
 * solver handle bound to the SAME stream. Nothing else synchronizes.
 */

module;

#include "nnls_bridge.h"

export module calaman.nnls;

import std;
import wwr.blas;            // wwrblasHandle_t, wwrblasStatus_t, WWRBLAS_*, wwrblasGetStream
import wwr.solver;          // wwrsolverDnHandle_t, WWRSOLVER_STATUS_SUCCESS
import wwr.runtime_api;     // wwrStream_t, wwrMemcpy(Async)/wwrMemsetAsync, wwrSuccess
import wwr.wrappers.blas;   // gemv, trsv, nrm2
import wwr.wrappers.solver; // ormqr, ormqr_bufferSize
import calaman.common;      // WorkspaceLayout, carve_workspace
import calaman.geqp3;       // geqp3, geqp3_work_size
export import calaman.iterative; // IterationInfo, stop_reason, converged

namespace calaman {

// ========================================================================
// Internal workspace (not exported): nnls() is instantiated in this module's
// instantiations.cpp, so importers link the symbol and never see these.
// ========================================================================

namespace nnls_detail {

/// @brief Length (T elements) of ormqr's workspace for the worst case (p == n)
///
/// Queried at the Q^T-b maximum: side Left, op transpose, an m-by-1 rhs with
/// k = min(m,n) reflectors. Returns 1 on any query failure or empty A.
template<typename T>
int ormqr_work_len(wwr::wwrsolverDnHandle_t solver, const int m, const int n) {
  if (m < 1 || n < 1) {
    return 1;
  }
  const int k = m < n ? m : n;
  const T *dummy = nullptr;
  int lwork = 0;
  if (wwr::ormqr_bufferSize<T>(solver, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_OP_T, m, 1, k, dummy, m,
                               dummy, dummy, m, &lwork) != wwr::WWRSOLVER_STATUS_SUCCESS) {
    return 1;
  }
  return lwork < 1 ? 1 : lwork;
}

/// @brief Device workspace regions carved from the caller's buffer
template<typename T>
struct NnlsWorkspace {
  int *order = nullptr;         ///< P/Z partition: order[0:p) = P, order[p:n) = Z
  int *order_scratch = nullptr; ///< scratch for the P/Z compaction, n ints
  int *jpvt0 = nullptr;         ///< geqp3's pivot (0-based) uploaded for the scatter, n ints

  T *r = nullptr;   ///< residual b - Ax, m
  T *w = nullptr;   ///< dual A^T r, n
  T *AP = nullptr;  ///< gathered then geqp3-factored passive columns, m-by-n
  T *tau = nullptr; ///< geqp3 reflector scalars, min(m,n)
  T *qb = nullptr;  ///< copy of b, becomes Q^T b then the pivoted-local z, m
  T *z = nullptr;   ///< full n-length candidate, zero off P
  T *vn1 = nullptr; ///< geqp3 partial-norm scratch, n
  T *vn2 = nullptr; ///< geqp3 partial-norm scratch, n

  T *geqp3_work = nullptr;        ///< geqp3's own workspace on the P subproblem
  std::size_t geqp3_work_len = 0; ///< its length in T elements

  int *rank = nullptr;       ///< numerical rank of the P subproblem
  int *arg_pos = nullptr;    ///< masked-argmax result: position within order
  int *infeasible = nullptr; ///< masked-min-ratio result: any z_j <= 0 on P
  int *new_p = nullptr;      ///< P/Z compaction result
  int *devInfo = nullptr;    ///< ormqr's devInfo

  T *max_val = nullptr; ///< masked-argmax result: the max value
  T *alpha = nullptr;   ///< masked-min-ratio result: the step length

  T *ormqr_work = nullptr; ///< workspace for ormqr on the P subproblem
  int ormqr_lwork = 0;

  /// @brief Lay the regions out from @p layout -- the ONLY description of the
  ///        layout, run for sizing and carving alike (see carve_workspace).
  ///        Sized for the worst case (a fully passive solution, p == n).
  void carve(WorkspaceLayout &layout, const int m, const int n, const std::size_t geqp3_len,
             const int ormqr_len) {
    const std::size_t rows = m < 1 ? 1 : static_cast<std::size_t>(m);
    const std::size_t cols = n < 1 ? 1 : static_cast<std::size_t>(n);
    const int k = m < n ? m : n;
    const std::size_t kk = k < 1 ? 1 : static_cast<std::size_t>(k);

    order = layout.fixed<int>(cols);
    order_scratch = layout.fixed<int>(cols);
    jpvt0 = layout.fixed<int>(cols);
    r = layout.fixed<T>(rows);
    w = layout.fixed<T>(cols);
    AP = layout.fixed<T>(rows * cols);
    tau = layout.fixed<T>(kk);
    qb = layout.fixed<T>(rows);
    z = layout.fixed<T>(cols);
    vn1 = layout.fixed<T>(cols);
    vn2 = layout.fixed<T>(cols);

    geqp3_work_len = geqp3_len;
    geqp3_work = layout.fixed<T>(geqp3_len);

    int *const ints = layout.fixed<int>(5);
    rank = ints;
    if (ints != nullptr) {
      arg_pos = ints + 1;
      infeasible = ints + 2;
      new_p = ints + 3;
      devInfo = ints + 4;
    }

    T *const scalars = layout.fixed<T>(2);
    max_val = scalars;
    if (scalars != nullptr) {
      alpha = scalars + 1;
    }

    ormqr_lwork = ormqr_len;
    ormqr_work = layout.fixed<T>(static_cast<std::size_t>(ormqr_len));
  }
};

} // namespace nnls_detail

// ========================================================================
// Options / result reporting (exported)
// ========================================================================

/// @brief Why nnls() stopped
export enum class NnlsStopReason {
  Converged,        ///< Z empty, or max_{j in Z} w_j <= tol
  MaxIterations,    ///< the outer iteration budget ran out
  NumericalFailure, ///< a BLAS/solver/geqp3 call failed, or the inner guard tripped
};
static_assert(stop_reason<NnlsStopReason>);

/// @brief Tuning for nnls()
export template<typename T>
struct NnlsOptions {
  /// Stop when max_{j in Z} w_j <= tol; also the "has x_j reached zero"
  /// threshold the inner loop uses when moving entries back to Z.
  T tol = T(10) * std::numeric_limits<T>::epsilon();

  /// Outer iteration cap. Zero uses scipy's default, 3 * n.
  int max_iterations = 0;
};

/// @brief What nnls() did
export template<typename T>
struct NnlsInfo : IterationInfo<NnlsStopReason> {
  T residual_norm = T(0); ///< final ||b - Ax||_2
};

// ========================================================================
// Workspace size (exported)
// ========================================================================

/// @brief Device workspace nnls() needs, in bytes, sized for the worst case
///        (a fully passive solution, p == n)
///
/// @tparam T     Element type (float or double)
/// @param solver Solver handle, used only to size ormqr's workspace
/// @param m      Rows of A
/// @param n      Columns of A
export template<typename T>
std::size_t nnls_bufferSize(wwr::wwrsolverDnHandle_t solver, const int m, const int n) {
  return carve_workspace<nnls_detail::NnlsWorkspace<T>>(
      nullptr, nullptr, m, n, geqp3_work_size(m, n), nnls_detail::ormqr_work_len<T>(solver, m, n));
}

// ========================================================================
// Solver (exported)
// ========================================================================

/// @brief Non-negative least squares: min ||A x - b||_2 subject to x >= 0
///
/// Produces x >= 0 in @p x. @p A (m-by-n, column-major, leading dimension
/// @p lda) and @p b (m) are not modified. @p work is a 256-byte-aligned device
/// buffer of at least nnls_bufferSize<T>(@p solver, m, n) bytes.
///
/// Returns WWRBLAS_STATUS_SUCCESS whenever the iteration reached a stopping
/// condition -- including hitting @p opts.max_iterations, which is an outcome,
/// not an error; read @p info to tell Converged from MaxIterations.
/// WWRBLAS_STATUS_INVALID_VALUE for bad dimensions, null pointers, or an
/// undersized buffer; any other status is a numerical failure (see @p info).
///
/// @tparam T       Element type (float or double)
/// @param blas     Blas handle in host pointer mode; its stream carries every op
/// @param solver   Solver handle bound to the SAME stream as @p blas
/// @param m        Rows of A
/// @param n        Columns of A
/// @param A        m-by-n column-major matrix, not modified
/// @param lda      Leading dimension of A (>= m)
/// @param b        Right-hand side, m
/// @param x        Out: solution, n, x >= 0
/// @param work     Device workspace, 256-byte aligned
/// @param work_bytes Size of @p work, >= nnls_bufferSize<T>(solver, m, n)
/// @param opts     Tuning
/// @param info     Host out, may be null
export template<typename T>
wwr::wwrblasStatus_t nnls(wwr::wwrblasHandle_t blas, wwr::wwrsolverDnHandle_t solver, const int m,
                          const int n, const T *A, const int lda, const T *b, T *x, void *work,
                          const std::size_t work_bytes, const NnlsOptions<T> &opts = {},
                          NnlsInfo<T> *info = nullptr) {
  if (m < 1 || n < 1 || lda < m) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  if (A == nullptr || b == nullptr || x == nullptr || work == nullptr) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  // One carve serves the undersized-buffer check and the region pointers.
  nnls_detail::NnlsWorkspace<T> ws;
  if (work_bytes < carve_workspace(work, &ws, m, n, geqp3_work_size(m, n),
                                   nnls_detail::ormqr_work_len<T>(solver, m, n))) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  wwr::wwrStream_t stream{};
  if (wwr::wwrblasGetStream(blas, &stream) != wwr::WWRBLAS_STATUS_SUCCESS) {
    return wwr::WWRBLAS_STATUS_NOT_INITIALIZED;
  }

  NnlsInfo<T> local{};
  const auto publish = [&](const NnlsStopReason reason, const wwr::wwrblasStatus_t st) {
    local.reason = reason;
    if (info != nullptr) {
      *info = local;
    }
    return st;
  };

  const int max_iter = opts.max_iterations > 0 ? opts.max_iterations : 3 * n;
  const T rank_rel_tol = static_cast<T>(m < n ? m : n) * std::numeric_limits<T>::epsilon();
  const T one = T{1};
  const T zero = T{0};
  const T neg_one = T{-1};
  const std::size_t mbytes = static_cast<std::size_t>(m) * sizeof(T);
  const std::size_t nbytes = static_cast<std::size_t>(n) * sizeof(T);

  std::vector<int> jpvt(static_cast<std::size_t>(n));
  std::vector<int> jpvt0(static_cast<std::size_t>(n));

  device::nnls_init<T>(stream, ws.order, x, n);
  int p = 0;

  for (int iter = 0; iter < max_iter; ++iter) {
    // ── step 1: w = A^T(b - Ax) ────────────────────────────────────────
    if (wwr::wwrMemcpyAsync(ws.r, b, mbytes, wwr::wwrMemcpyDeviceToDevice, stream) !=
        wwr::wwrSuccess) {
      return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
    }
    if (wwr::gemv<T, int>(blas, wwr::WWRBLAS_OP_N, m, n, &neg_one, A, lda, x, 1, &one, ws.r, 1) !=
        wwr::WWRBLAS_STATUS_SUCCESS) {
      return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
    }
    if (wwr::gemv<T, int>(blas, wwr::WWRBLAS_OP_T, m, n, &one, A, lda, ws.r, 1, &zero, ws.w, 1) !=
        wwr::WWRBLAS_STATUS_SUCCESS) {
      return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
    }

    // ── step 2: stopping test ──────────────────────────────────────────
    bool converged = (p >= n);
    int host_arg_pos = 0;
    if (!converged) {
      device::nnls_masked_argmax<T>(stream, ws.w, ws.order, p, n, ws.max_val, ws.arg_pos);
      T host_max_val{};
      if (wwr::wwrMemcpyAsync(&host_max_val, ws.max_val, sizeof(T), wwr::wwrMemcpyDeviceToHost,
                              stream) != wwr::wwrSuccess ||
          wwr::wwrMemcpyAsync(&host_arg_pos, ws.arg_pos, sizeof(int), wwr::wwrMemcpyDeviceToHost,
                              stream) != wwr::wwrSuccess ||
          wwr::wwrStreamSynchronize(stream) != wwr::wwrSuccess) {
        return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
      }
      converged = host_max_val <= opts.tol;
    }
    if (converged) {
      local.iterations = iter;
      if (wwr::nrm2<T, int>(blas, m, ws.r, 1, &local.residual_norm) !=
          wwr::WWRBLAS_STATUS_SUCCESS) {
        return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
      }
      return publish(NnlsStopReason::Converged, wwr::WWRBLAS_STATUS_SUCCESS);
    }

    // ── step 3: move the entering column from Z to P ───────────────────
    device::nnls_swap_order(stream, ws.order, p, host_arg_pos);
    ++p;

    // ── step 4: resolve the P subproblem, feasibly ─────────────────────
    // A line step can zero every passive entry at once (a tie among columns
    // crossing zero together), emptying P; z is then the zero vector, so p == 0
    // is checked before every subproblem solve, not just the first.
    bool feasible = false;
    for (int inner = 0; inner < n + 1; ++inner) {
      if (p == 0) {
        if (wwr::wwrMemsetAsync(x, 0, nbytes, stream) != wwr::wwrSuccess) {
          return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
        }
        feasible = true;
        break;
      }

      // Gather the passive columns and pivoted-QR factor them. geqp3 overwrites
      // AP with R + reflectors (LAPACK layout ormqr reads as-is), fills tau and
      // the 1-based host pivot jpvt[0:p).
      device::nnls_gather_columns<T>(stream, ws.AP, m, A, lda, ws.order, p);
      if (wwr::wwrMemcpyAsync(ws.qb, b, mbytes, wwr::wwrMemcpyDeviceToDevice, stream) !=
          wwr::wwrSuccess) {
        return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
      }
      if (geqp3<T>(blas, m, p, ws.AP, m, jpvt.data(), ws.tau, ws.vn1, ws.vn2, ws.geqp3_work) !=
          wwr::WWRBLAS_STATUS_SUCCESS) {
        return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
      }

      const int kref = m < p ? m : p; // reflectors / R diagonal length
      device::nnls_rank<T>(stream, ws.AP, m, kref, rank_rel_tol, ws.rank);
      int rank = 0;
      if (wwr::wwrMemcpyAsync(&rank, ws.rank, sizeof(int), wwr::wwrMemcpyDeviceToHost, stream) !=
              wwr::wwrSuccess ||
          wwr::wwrStreamSynchronize(stream) != wwr::wwrSuccess) {
        return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
      }

      // qb := Q^T b, then y := R(0:rank,0:rank)^{-1} qb(0:rank); the deficient
      // tail qb(rank:p) is zeroed -- the minimum-norm choice.
      if (wwr::ormqr<T>(solver, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_OP_T, m, 1, kref, ws.AP, m,
                        ws.tau, ws.qb, m, ws.ormqr_work, ws.ormqr_lwork, ws.devInfo) !=
          wwr::WWRSOLVER_STATUS_SUCCESS) {
        return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
      }
      if (rank > 0 &&
          wwr::trsv<T, int>(blas, wwr::WWRBLAS_FILL_MODE_UPPER, wwr::WWRBLAS_OP_N,
                            wwr::WWRBLAS_DIAG_NON_UNIT, rank, ws.AP, m, ws.qb, 1) !=
              wwr::WWRBLAS_STATUS_SUCCESS) {
        return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
      }
      if (rank < p &&
          wwr::wwrMemsetAsync(ws.qb + rank, 0, static_cast<std::size_t>(p - rank) * sizeof(T),
                              stream) != wwr::wwrSuccess) {
        return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
      }

      // Scatter z[order[jpvt0[j]]] = qb[j]: compose geqp3's pivot (0-based) with
      // the P/Z partition. z is zeroed first so entries off P stay 0.
      for (int j = 0; j < p; ++j) {
        jpvt0[static_cast<std::size_t>(j)] = jpvt[static_cast<std::size_t>(j)] - 1;
      }
      if (wwr::wwrMemcpyAsync(ws.jpvt0, jpvt0.data(), static_cast<std::size_t>(p) * sizeof(int),
                              wwr::wwrMemcpyHostToDevice, stream) != wwr::wwrSuccess) {
        return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
      }
      if (wwr::wwrMemsetAsync(ws.z, 0, nbytes, stream) != wwr::wwrSuccess) {
        return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
      }
      device::nnls_scatter_z<T>(stream, ws.z, ws.qb, ws.order, ws.jpvt0, p);

      device::nnls_masked_min_ratio<T>(stream, x, ws.z, ws.order, p, ws.alpha, ws.infeasible);
      T host_alpha{};
      int host_infeasible = 0;
      if (wwr::wwrMemcpyAsync(&host_alpha, ws.alpha, sizeof(T), wwr::wwrMemcpyDeviceToHost,
                              stream) != wwr::wwrSuccess ||
          wwr::wwrMemcpyAsync(&host_infeasible, ws.infeasible, sizeof(int),
                              wwr::wwrMemcpyDeviceToHost, stream) != wwr::wwrSuccess ||
          wwr::wwrStreamSynchronize(stream) != wwr::wwrSuccess) {
        return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
      }

      if (host_infeasible == 0) {
        if (wwr::wwrMemcpyAsync(x, ws.z, nbytes, wwr::wwrMemcpyDeviceToDevice, stream) !=
            wwr::wwrSuccess) {
          return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
        }
        feasible = true;
        break;
      }

      device::nnls_line_step<T>(stream, x, ws.z, host_alpha, n);
      device::nnls_compact_zeros<T>(stream, ws.order, ws.order_scratch, x, p, opts.tol, ws.new_p);
      int new_p = p;
      if (wwr::wwrMemcpyAsync(&new_p, ws.new_p, sizeof(int), wwr::wwrMemcpyDeviceToHost, stream) !=
              wwr::wwrSuccess ||
          wwr::wwrStreamSynchronize(stream) != wwr::wwrSuccess) {
        return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
      }
      p = new_p;
    }
    if (!feasible) {
      return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
    }

    local.iterations = iter + 1;
  }

  if (wwr::nrm2<T, int>(blas, m, ws.r, 1, &local.residual_norm) != wwr::WWRBLAS_STATUS_SUCCESS) {
    return publish(NnlsStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
  }
  return publish(NnlsStopReason::MaxIterations, wwr::WWRBLAS_STATUS_SUCCESS);
}

// Instantiated in instantiations.cpp; importers link these, never instantiate.
extern template std::size_t nnls_bufferSize<float>(wwr::wwrsolverDnHandle_t, const int, const int);
extern template std::size_t nnls_bufferSize<double>(wwr::wwrsolverDnHandle_t, const int, const int);

extern template wwr::wwrblasStatus_t
nnls<float>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, const int, const int, const float *,
            const int, const float *, float *, void *, const std::size_t, const NnlsOptions<float> &,
            NnlsInfo<float> *);
extern template wwr::wwrblasStatus_t
nnls<double>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, const int, const int, const double *,
             const int, const double *, double *, void *, const std::size_t,
             const NnlsOptions<double> &, NnlsInfo<double> *);

} // namespace calaman
