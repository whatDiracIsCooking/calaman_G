/**
 * @file pade.cppm
 * @brief The [m/m] diagonal Pade approximant r_m(A) = q_m(A)^-1 p_m(A), unscaled
 *
 * The :pade partition of calaman.expm -- the unscaled kernel of the algorithm.
 *
 * q_m(x) = p_m(-x), so splitting p_m into its even and odd halves gives
 * p_m(A) = V + U and q_m(A) = V - U, and a single LU solve of (V - U) X = (V + U)
 * produces r_m(A). THE SIGN FLIP IS THE WHOLE TRICK: the denominator costs
 * nothing beyond the elementwise pass that already builds the numerator -- a
 * general polynomial evaluator (calaman.horner) cannot see that relationship.
 * The README has the degree-13 nested form and the cost table.
 *
 * Accurate only while norm_1(A) stays below pade_theta<T>(m); use expm() for a
 * general matrix. Unlike expm() it enqueues work without ever synchronizing.
 */

module;

#include "expm_bridge.h"
// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import supplies.
#include "error_handling/error_macros.h"

export module calaman.expm:pade;

import std;
import wwr.runtime_api;     // wwrStream_t
import wwr.blas;            // wwrblasHandle_t, WWRBLAS_*, pointer-mode get/set
import wwr.solver;          // wwrsolverDnHandle_t
import wwr.complex;         // wwrFloatComplex, wwrDoubleComplex (the extern-template list)
import wwr.wrappers.common; // usual_fp, ComplexToRealType
import wwr.wrappers.blas;   // gemm
import wwr.wrappers.solver; // getrf, getrf_bufferSize, getrs
import wwr.extension.blas;  // ScopedPointerMode (forces host mode for the gemms)
import calaman.common;      // WorkspaceLayout
import calaman.error_handling; // Status, PointerModeStatus
import :detail;             // as_element, kMaxDim, PadeWorkspace

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declaration below sits in the plain namespace.

/**
 * @brief Evaluate the [m/m] diagonal Pade approximant r_m(A) = q_m(A)^-1 p_m(A).
 *
 * The unscaled kernel of the algorithm: accurate only while norm_1(A) stays
 * below pade_theta<T>(m); use expm() for a general matrix. Splits p_m into even
 * and odd halves so one LU solve of (V - U) X = (V + U) finishes the job (see
 * the file header and the README). Cost: pade_num_gemms(m) matrix products plus
 * one or two fused kernels, one getrf and one getrs. Unlike expm() it enqueues
 * work without ever synchronizing.
 *
 * @tparam T Element type (one of the four usual_fp types)
 * @param cublas_handle   BLAS handle, bound to @p stream, default (host) pointer mode.
 * @param cusolver_handle Solver handle, bound to @p stream.
 * @param stream          Stream the custom kernels are enqueued on.
 * @param m               Pade degree; one of 3, 5, 7, 9, 13.
 * @param n               Matrix dimension.
 * @param d_A             Input matrix, column-major, n x n, leading dimension @p lda.
 * @param lda             Leading dimension of @p d_A (>= n).
 * @param d_r             Output r_m(A), column-major, leading dimension @p ldr.
 * @param ldr             Leading dimension of @p d_r (>= n).
 * @param d_work          Device workspace of at least pade_bufferSize<T>(m, n) bytes.
 * @param lwork_bytes     Size of @p d_work in bytes.
 * @param d_info          Device array of 2 ints: [0] getrf info, [1] getrs info.
 * @return Status: SUCCESS, or the first error; a BLAS or solver failure carries
 *         its own domain's status, a bad argument WWRBLAS_STATUS_INVALID_VALUE,
 *         an undersized workspace WWRBLAS_STATUS_ALLOC_FAILED.
 *
 * @pre @p d_A and @p d_r must not overlap, and neither may overlap @p d_work.
 * @pre Both handles must be bound to @p stream.
 * @post The pointer mode of @p cublas_handle is left as it was found.
 */
export template<wwr::usual_fp T>
Status pade(wwr::wwrblasHandle_t cublas_handle, wwr::wwrsolverDnHandle_t cusolver_handle,
            wwr::wwrStream_t stream, const int m, const int n, const T *d_A, const int lda, T *d_r,
            const int ldr, void *d_work, const std::size_t lwork_bytes, int *d_info) {
  using RealT = wwr::ComplexToRealType<T>;

  const double *b = pade_coeffs(m);
  if (b == nullptr || n < 1 || n > kMaxDim || lda < n || ldr < n || d_work == nullptr) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  int lwork_getrf = 0;
  CLM_TRY(wwr::getrf_bufferSize<T>(cusolver_handle, n, n, static_cast<T *>(nullptr), n,
                                   &lwork_getrf));

  // One carve, sizing and laying out together: total() is the requirement, and
  // the pointers are unused until after the lwork check below passes.
  WorkspaceLayout layout(d_work);
  PadeWorkspace<T> ws;
  ws.carve(layout, n, m, lwork_getrf);
  if (lwork_bytes < layout.total()) {
    return wwr::WWRBLAS_STATUS_ALLOC_FAILED;
  }
  const int np = pade_num_powers(m);

  // The gemms use host scalars, so the handle must be in HOST pointer mode
  // regardless of how the caller left it. ScopedPointerMode records the entry
  // mode and restores it from its destructor -- which fires on CLM_TRY's bare
  // `return` early exits too, honoring the @post below. PointerModeStatus folds a
  // failing get/set/restore into pm_status (calaman ships no abort policy in
  // src/); CLM_TRY surfaces the entering get/set before the body relies on it.
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode scope{cublas_handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                                PointerModeStatus{&pm_status}};
  CLM_TRY(pm_status);

  const T one = as_element<T>(RealT{1});
  const T zero = as_element<T>(RealT{0});

  // Z = X * Y, and Z += X * Y (the beta = 1 that makes the degree-13 tail free)
  const auto matmul = [&](const T *X, const int ldx, const T *Y, const int ldy, T *Z,
                          const int ldz) {
    return wwr::gemm<T>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &one, X, ldx,
                        Y, ldy, &zero, Z, ldz);
  };
  const auto matmul_acc = [&](const T *X, const int ldx, const T *Y, const int ldy, T *Z,
                              const int ldz) {
    return wwr::gemm<T>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &one, X, ldx,
                        Y, ldy, &one, Z, ldz);
  };

  // ── Powers of A: A2, then A4, A6, A8 as far as the degree needs ────
  CLM_TRY(matmul(d_A, lda, d_A, lda, ws.P[0], n));
  if (np >= 2) {
    CLM_TRY(matmul(ws.P[0], n, ws.P[0], n, ws.P[1], n));
  }
  if (np >= 3) {
    CLM_TRY(matmul(ws.P[1], n, ws.P[0], n, ws.P[2], n));
  }
  if (np >= 4) {
    CLM_TRY(matmul(ws.P[1], n, ws.P[1], n, ws.P[3], n));
  }

  RealT cv[5] = {};
  RealT cw[5] = {};

  if (m != 13) {
    // ── Even and odd halves of the numerator, in one pass ──────────
    for (int i = 0; i <= np; ++i) {
      cv[i] = static_cast<RealT>(b[2 * i]);
      cw[i] = static_cast<RealT>(b[2 * i + 1]);
    }
    device::pade_even_odd<T, RealT>(stream, n, np, ws.P[0], ws.P[1], ws.P[2], ws.P[3], cv, cw, ws.V,
                                    ws.W);
  } else {
    // ── Odd parity: X = b9*A2 + b11*A4 + b13*A6, W = the tail ──────
    cv[0] = RealT{0};
    cv[1] = static_cast<RealT>(b[9]);
    cv[2] = static_cast<RealT>(b[11]);
    cv[3] = static_cast<RealT>(b[13]);
    cw[0] = static_cast<RealT>(b[1]);
    cw[1] = static_cast<RealT>(b[3]);
    cw[2] = static_cast<RealT>(b[5]);
    cw[3] = static_cast<RealT>(b[7]);
    device::pade_even_odd<T, RealT>(stream, n, 3, ws.P[0], ws.P[1], ws.P[2],
                                    static_cast<const T *>(nullptr), cv, cw, ws.X, ws.W);

    CLM_TRY(matmul_acc(ws.P[2], n, ws.X, n, ws.W, n)); // W += A6 * X

    // ── Even parity, reusing X now that the odd side is finished ───
    cv[0] = RealT{0};
    cv[1] = static_cast<RealT>(b[8]);
    cv[2] = static_cast<RealT>(b[10]);
    cv[3] = static_cast<RealT>(b[12]);
    cw[0] = static_cast<RealT>(b[0]);
    cw[1] = static_cast<RealT>(b[2]);
    cw[2] = static_cast<RealT>(b[4]);
    cw[3] = static_cast<RealT>(b[6]);
    device::pade_even_odd<T, RealT>(stream, n, 3, ws.P[0], ws.P[1], ws.P[2],
                                    static_cast<const T *>(nullptr), cv, cw, ws.X, ws.V);

    CLM_TRY(matmul_acc(ws.P[2], n, ws.X, n, ws.V, n)); // V += A6 * X
  }

  // ── U = A * W. Safe in place of P[0]: the powers are dead from here ─
  CLM_TRY(matmul(d_A, lda, ws.W, n, ws.U, n));

  // ── p(A) = V + U into the output, q(A) = V - U over V, in one pass ─
  device::pade_split<T>(stream, n, ws.U, ws.V, d_r, ldr, ws.Q);

  // ── Solve q(A) X = p(A) ────────────────────────────────────────────
  CLM_TRY(wwr::getrf<T>(cusolver_handle, n, n, ws.Q, n, ws.work_getrf, ws.ipiv, d_info));
  CLM_TRY(wwr::getrs<T>(cusolver_handle, wwr::WWRBLAS_OP_N, n, n, ws.Q, n, ws.ipiv, d_r, ldr,
                        d_info + 1));

  return wwr::WWRBLAS_STATUS_SUCCESS;
}

// Instantiated once in instantiations.cpp -- its body names the .cu-side
// launchers declared only in the global module fragment, so an importer never
// re-instantiates it.
extern template Status pade<float>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, wwr::wwrStream_t,
                                   int, int, const float *, int, float *, int, void *, std::size_t,
                                   int *);
extern template Status pade<double>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                    wwr::wwrStream_t, int, int, const double *, int, double *, int,
                                    void *, std::size_t, int *);
extern template Status pade<wwr::wwrFloatComplex>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                                  wwr::wwrStream_t, int, int,
                                                  const wwr::wwrFloatComplex *, int,
                                                  wwr::wwrFloatComplex *, int, void *, std::size_t,
                                                  int *);
extern template Status pade<wwr::wwrDoubleComplex>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                                   wwr::wwrStream_t, int, int,
                                                   const wwr::wwrDoubleComplex *, int,
                                                   wwr::wwrDoubleComplex *, int, void *,
                                                   std::size_t, int *);

} // namespace calaman
