/**
 * @file expm.cppm
 * @brief exp(A) by scaling and squaring with a diagonal Pade approximant
 *        (Higham's Algorithm 2.3)
 *
 * The :expm partition of calaman.expm -- the driver that composes the rest.
 *
 *   1. Take the CHEAPEST degree m in {3, 5, 7, 9, 13} whose backward-error
 *      threshold theta_m already covers norm_1(A)              (:plan, :norm1).
 *   2. Only if even theta_13 does not, SCALE: pick the smallest s with
 *      norm_1(A)/2^s <= theta_13, evaluate X = r_13(A/2^s) (:pade), and square
 *      s times.
 *
 * NO BALANCING. expm does not balance the matrix itself: balancing is the
 * orthogonal similarity exp(A) = D exp(D^-1 A D) D^-1, which composes around this
 * routine rather than belonging inside it. A caller who wants it runs
 * calaman.gebal (GebalJob::Scale) to form D^-1 A D, calls expm on that, and
 * re-wraps the result with D and D^-1 (the README spells out the recipe).
 *
 * SYNCHRONIZES @p stream once, to read norm_1(A) onto the host (the degree and
 * the scaling exponent drive host-side control flow). Call pade() directly when
 * the norm is known a priori and a fully asynchronous path is required.
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import supplies.
#include "error_handling/error_macros.h"

export module calaman.expm:expm;

import std;
import wwr.runtime_api;     // wwrStream_t
import wwr.blas;            // wwrblasHandle_t, WWRBLAS_*, pointer-mode get/set
import wwr.solver;          // wwrsolverDnHandle_t
import wwr.complex;         // wwrFloatComplex, wwrDoubleComplex (the extern-template list)
import wwr.wrappers.common; // usual_fp, ComplexToRealType
import wwr.wrappers.blas;   // gemm, geam
import wwr.extension.blas;  // ScopedPointerMode (forces host mode for the gemms/geams)
import calaman.common;      // carve_workspace
import calaman.error_handling; // Status, PointerModeStatus
import :detail;             // as_element, kMaxDim, ExpmWorkspace
import :plan;               // ExpmPlan, expm_plan
import :norm1;              // matrix_norm1
import :buffer_size;        // pade_bufferSize, expm_bufferSize
import :pade;               // pade

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declaration below sits in the plain namespace.

/**
 * @brief Matrix exponential by scaling and squaring with a diagonal Pade approximant.
 *
 * Higham's Algorithm 2.3 (see the file header). Total cost is
 * pade_num_gemms(m) + s matrix products; expm_plan() reports what a given norm
 * will cost before the call.
 *
 * @tparam T Element type (one of the four usual_fp types)
 * @param cublas_handle   BLAS handle, bound to @p stream, default (host) pointer mode.
 * @param cusolver_handle Solver handle, bound to @p stream.
 * @param stream          Stream all work is enqueued on.
 * @param n               Matrix dimension.
 * @param d_A             Input matrix, column-major, n x n, leading dimension @p lda.
 * @param lda             Leading dimension of @p d_A (>= n).
 * @param d_expA          Output exp(A), column-major, leading dimension @p lde.
 * @param lde             Leading dimension of @p d_expA (>= n).
 * @param d_work          Device workspace of at least expm_bufferSize<T>() bytes.
 * @param lwork_bytes     Size of @p d_work in bytes.
 * @param d_info          Device array of 2 ints: [0] getrf info, [1] getrs info.
 * @param plan            Optional output: the plan expm executed -- degree,
 *                        squarings and gemm count, the same ExpmPlan expm_plan()
 *                        would have predicted from norm_1(A).
 * @return Status: SUCCESS, or the first error; a BLAS or solver failure carries
 *         its own domain's status, a bad argument (including a 1-norm that is not
 *         finite) WWRBLAS_STATUS_INVALID_VALUE, an undersized workspace
 *         WWRBLAS_STATUS_ALLOC_FAILED.
 *
 * @pre @p d_A and @p d_expA must not overlap.
 * @pre Both handles must be bound to @p stream.
 * @post The pointer mode of @p cublas_handle is left as it was found.
 * @warning Synchronizes @p stream (see the file header).
 */
export template<wwr::usual_fp T>
Status expm(wwr::wwrblasHandle_t cublas_handle, wwr::wwrsolverDnHandle_t cusolver_handle,
            wwr::wwrStream_t stream, const int n, const T *d_A, const int lda, T *d_expA,
            const int lde, void *d_work, const std::size_t lwork_bytes, int *d_info,
            ExpmPlan *plan = nullptr) {
  using RealT = wwr::ComplexToRealType<T>;

  if (n < 1 || n > kMaxDim || lda < n || lde < n || d_work == nullptr) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  std::size_t required = 0;
  CLM_TRY(expm_bufferSize<T>(cusolver_handle, n, &required));
  if (lwork_bytes < required) {
    return wwr::WWRBLAS_STATUS_ALLOC_FAILED;
  }

  ExpmWorkspace<T> ws;
  carve_workspace(d_work, &ws, n); // size already checked against expm_bufferSize above

  // HOST pointer mode for the host-scalar gemms/geams, restored on every exit.
  // ScopedPointerMode's destructor restores on CLM_TRY's bare-`return` early paths
  // too (honoring the @post); PointerModeStatus folds a failing get/set/restore
  // into pm_status (calaman ships no abort policy in src/), and CLM_TRY surfaces
  // the entering get/set before the body relies on host mode.
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode scope{cublas_handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                                PointerModeStatus{&pm_status}};
  CLM_TRY(pm_status);

  const T one = as_element<T>(RealT{1});
  const T zero = as_element<T>(RealT{0});

  // ── The norm the whole plan hangs off ──────────────────────────────
  const RealT norm = matrix_norm1<T>(stream, n, d_A, lda, ws.colsum);
  if (!std::isfinite(norm)) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  const ExpmPlan chosen = expm_plan<T>(norm);
  if (plan != nullptr) {
    *plan = chosen;
  }

  std::size_t pade_bytes = 0;
  CLM_TRY(pade_bufferSize<T>(cusolver_handle, chosen.m, n, &pade_bytes));

  // ── Pick what actually gets exponentiated ──────────────────────────
  const T *src = d_A;
  int ld_src = lda;

  if (chosen.s > 0) {
    const T factor = as_element<T>(std::ldexp(RealT{1}, -chosen.s));
    CLM_TRY(wwr::geam<T>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, &factor, d_A,
                         lda, &zero, d_A, lda, ws.As, n));
    src = ws.As;
    ld_src = n;
  }

  CLM_TRY(pade<T>(cublas_handle, cusolver_handle, stream, chosen.m, n, src, ld_src, d_expA, lde,
                  ws.pade_base, pade_bytes, d_info));

  // ── Square s times, ping-ponging between the output and the scratch ─
  //
  // ws.sq aliases ws.As, which pade() has finished reading by now. That is also
  // why the squaring cannot start in the scratch to save the copy below the way
  // horner() does: the scratch is still the input at the moment pade() would
  // write into it. One block of n^2 is worth more than one geam.
  T *cur = d_expA;
  int ld_cur = lde;
  T *other = ws.sq;
  int ld_other = n;

  for (int k = 0; k < chosen.s; ++k) {
    CLM_TRY(wwr::gemm<T>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &one, cur,
                         ld_cur, cur, ld_cur, &zero, other, ld_other));
    std::swap(cur, other);
    std::swap(ld_cur, ld_other);
  }

  // An odd number of squarings leaves the result in the scratch block.
  if (cur != d_expA) {
    CLM_TRY(wwr::geam<T>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, &one, cur,
                         ld_cur, &zero, cur, ld_cur, d_expA, lde));
  }

  return wwr::WWRBLAS_STATUS_SUCCESS;
}

// Instantiated once in instantiations.cpp -- its body names the .cu-side
// launchers (transitively, through pade()) declared only in a global module
// fragment, so an importer never re-instantiates it.
extern template Status expm<float>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, wwr::wwrStream_t,
                                   int, const float *, int, float *, int, void *, std::size_t,
                                   int *, ExpmPlan *);
extern template Status expm<double>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                    wwr::wwrStream_t, int, const double *, int, double *, int,
                                    void *, std::size_t, int *, ExpmPlan *);
extern template Status expm<wwr::wwrFloatComplex>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                                  wwr::wwrStream_t, int,
                                                  const wwr::wwrFloatComplex *, int,
                                                  wwr::wwrFloatComplex *, int, void *, std::size_t,
                                                  int *, ExpmPlan *);
extern template Status expm<wwr::wwrDoubleComplex>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                                   wwr::wwrStream_t, int,
                                                   const wwr::wwrDoubleComplex *, int,
                                                   wwr::wwrDoubleComplex *, int, void *,
                                                   std::size_t, int *, ExpmPlan *);

} // namespace calaman
