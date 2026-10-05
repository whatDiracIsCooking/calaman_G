/**
 * @file expm_herm.cppm
 * @brief exp(A) for a symmetric (real) or Hermitian (complex) A, by eigenvalue
 *        decomposition
 *
 * The :herm partition of calaman.expm -- a second, specialised driver that sits
 * beside the general scaling-and-squaring expm(). For a self-adjoint A the
 * spectral theorem gives A = U diag(w) U^H with U orthogonal/unitary and the
 * eigenvalues w REAL, so
 *
 *     exp(A) = U diag(exp(w)) U^H
 *
 * is exact up to the eigensolver's accuracy -- no Pade, no scaling, no squaring.
 * This is the standard and numerically preferred route for this matrix class;
 * the general expm() stays the right tool for a non-normal matrix.
 *
 * STRUCTURE. A host composition over the wrapped symmetric / Hermitian
 * eigensolver (syevd for real T, heevd for complex T -- chosen by if constexpr),
 * one fused kernel of its own (herm_exp_scale, in expm.cu: M = U diag(exp(w))),
 * and two wrapped BLAS calls (geam to stage the const input, gemm for the U^H
 * product -- transpose for real, conjugate-transpose for complex). Every name is
 * the backend-neutral wwr* one, so the single source builds for either vendor.
 *
 * SYNCHRONIZES nothing itself -- unlike the general expm(), which reads the
 * 1-norm back to the host. The eigenvalues stay on the device; the exp is taken
 * there. The caller's own stream sync is all that is needed to read exp(A) back.
 *
 * STATUS. Returns calaman::Status via CLM_TRY, like the rest of the module: a
 * solver failure (syevd/heevd non-convergence, reported in @p d_info) or a BLAS
 * failure carries its own domain's code; a bad argument is WWRBLAS_STATUS_INVALID_VALUE
 * and an undersized workspace WWRBLAS_STATUS_ALLOC_FAILED.
 */

module;

#include "expm_bridge.h"
// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import supplies.
#include "error_handling/error_macros.h"

export module calaman.expm:herm;

import std;
import wwr.runtime_api;     // wwrStream_t
import wwr.blas;            // wwrblasHandle_t, wwrblasFillMode_t, wwrblasOperation_t, WWRBLAS_*
import wwr.solver;          // wwrsolverDnHandle_t, WWRSOLVER_EIG_MODE_VECTOR
import wwr.complex;         // wwrFloatComplex, wwrDoubleComplex (the extern-template list)
import wwr.wrappers.blas;   // geam, gemm
import wwr.wrappers.solver; // syevd, heevd, and their *_bufferSize queries
import wwr.extension.blas;  // ScopedPointerMode (forces host mode for the geam/gemm)
import calaman.common;      // carve_workspace, usual_fp, ComplexToRealType
import calaman.error_handling; // Status, PointerModeStatus
import :detail;             // as_element, kMaxDim, HermWorkspace
// device::herm_exp_scale arrives by #include "expm_bridge.h" in the GMF above.

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so each template carries its own
// `export` and the declarations below sit in the plain namespace.

// Query the syevd/heevd scratch length (in elements of T) for an n x n problem.
// Non-exported: shared between the bufferSize query and the driver so the two
// cannot disagree on which eigensolver backs a given T. The if constexpr keeps
// syevd (real-only) and heevd (complex-only) each out of the other's branch.
template<calaman::usual_fp T>
Status herm_eig_lwork(wwr::wwrsolverDnHandle_t handle, const wwr::wwrblasFillMode_t uplo,
                      const int n, int *lwork) {
  using RealT = calaman::ComplexToRealType<T>;
  if constexpr (std::is_same_v<T, RealT>) {
    return wwr::syevd_bufferSize<T>(handle, wwr::WWRSOLVER_EIG_MODE_VECTOR, uplo, n,
                                    static_cast<const T *>(nullptr), n,
                                    static_cast<const RealT *>(nullptr), lwork);
  } else {
    return wwr::heevd_bufferSize<T>(handle, wwr::WWRSOLVER_EIG_MODE_VECTOR, uplo, n,
                                    static_cast<const T *>(nullptr), n,
                                    static_cast<const RealT *>(nullptr), lwork);
  }
}

/**
 * @brief Device workspace, in bytes, required by expm_herm().
 *
 * Two n*n blocks (the eigenvectors and their exp-scaled copy), the O(n)
 * eigenvalue vector and the eigensolver's own scratch -- far less than the Pade
 * ladder that expm_bufferSize reserves.
 *
 * @param handle      Solver handle (queried for the syevd/heevd workspace size).
 * @param uplo        Which triangle of A is read (matches the expm_herm call).
 * @param n           Matrix dimension.
 * @param lwork_bytes Output: required workspace in bytes.
 * @return Status: SUCCESS, WWRBLAS_STATUS_INVALID_VALUE for n < 1, or the
 *         solver-domain status if the eigensolver query fails.
 */
export template<calaman::usual_fp T>
Status expm_herm_bufferSize(wwr::wwrsolverDnHandle_t handle, const wwr::wwrblasFillMode_t uplo,
                            const int n, std::size_t *lwork_bytes) {
  if (n < 1) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  int lwork_eig = 0;
  CLM_TRY(herm_eig_lwork<T>(handle, uplo, n, &lwork_eig));

  // Null base: size only, from the carve expm_herm runs.
  *lwork_bytes = carve_workspace<HermWorkspace<T>>(nullptr, nullptr, n, lwork_eig);
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief exp(A) for a symmetric (real) or Hermitian (complex) A, by eigenvalue
 *        decomposition.
 *
 * Forms A = U diag(w) U^H with syevd (real) or heevd (complex), then
 * exp(A) = U diag(exp(w)) U^H -- one fused column scale and one gemm (see the
 * file header). Only the @p uplo triangle of A is read; the other is assumed to
 * mirror it. Exact up to the eigensolver's accuracy, so there is no degree or
 * scaling to report -- hence no plan output, unlike expm().
 *
 * @tparam T Element type (one of the four usual_fp types)
 * @param cublas_handle   BLAS handle, bound to @p stream, default (host) pointer mode.
 * @param cusolver_handle Solver handle, bound to @p stream.
 * @param stream          Stream the fused kernel is enqueued on.
 * @param uplo            WWRBLAS_FILL_MODE_UPPER / _LOWER: the triangle of A read.
 * @param n               Matrix dimension.
 * @param d_A             Input matrix, column-major, n x n, leading dimension @p lda;
 *                        symmetric (real) or Hermitian (complex) in the @p uplo triangle.
 * @param lda             Leading dimension of @p d_A (>= n).
 * @param d_expA          Output exp(A), column-major, leading dimension @p lde.
 * @param lde             Leading dimension of @p d_expA (>= n).
 * @param d_work          Device workspace of at least expm_herm_bufferSize<T>() bytes.
 * @param lwork_bytes     Size of @p d_work in bytes.
 * @param d_info          Device int: the eigensolver info code (0 on success, >0 if
 *                        the decomposition failed to converge).
 * @return Status: SUCCESS, or the first error; a solver or BLAS failure carries its
 *         own domain's status, a bad argument WWRBLAS_STATUS_INVALID_VALUE, an
 *         undersized workspace WWRBLAS_STATUS_ALLOC_FAILED.
 *
 * @pre @p d_A and @p d_expA must not overlap.
 * @pre Both handles must be bound to @p stream.
 * @post The pointer mode of @p cublas_handle is left as it was found.
 */
export template<calaman::usual_fp T>
Status expm_herm(wwr::wwrblasHandle_t cublas_handle, wwr::wwrsolverDnHandle_t cusolver_handle,
                 wwr::wwrStream_t stream, const wwr::wwrblasFillMode_t uplo, const int n,
                 const T *d_A, const int lda, T *d_expA, const int lde, void *d_work,
                 const std::size_t lwork_bytes, int *d_info) {
  using RealT = calaman::ComplexToRealType<T>;

  if (n < 1 || n > kMaxDim || lda < n || lde < n || d_work == nullptr) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  int lwork_eig = 0;
  CLM_TRY(herm_eig_lwork<T>(cusolver_handle, uplo, n, &lwork_eig));

  HermWorkspace<T> ws;
  if (lwork_bytes < carve_workspace(d_work, &ws, n, lwork_eig)) {
    return wwr::WWRBLAS_STATUS_ALLOC_FAILED;
  }

  // HOST pointer mode for the host-scalar geam/gemm, restored on every exit --
  // the same guard pade()/expm() use; ScopedPointerMode's destructor fires on
  // CLM_TRY's bare-`return` early paths too (honoring the @post), and
  // PointerModeStatus folds a failing get/set/restore into pm_status.
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode scope{cublas_handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                                PointerModeStatus{&pm_status}};
  CLM_TRY(pm_status);

  const T one = as_element<T>(RealT{1});
  const T zero = as_element<T>(RealT{0});

  // syevd/heevd overwrite their input with the eigenvectors, and d_A is const, so
  // copy it into the eigenvector block first. The copy is the full matrix; the
  // eigensolver only reads the uplo triangle, so the other half is immaterial.
  CLM_TRY(wwr::geam<T>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, &one, d_A, lda,
                       &zero, d_A, lda, ws.U, n));

  // A = U diag(w) U^H: eigenvalues ascending in ws.w, eigenvectors overwrite ws.U.
  if constexpr (std::is_same_v<T, RealT>) {
    CLM_TRY(wwr::syevd<T>(cusolver_handle, wwr::WWRSOLVER_EIG_MODE_VECTOR, uplo, n, ws.U, n, ws.w,
                          ws.eig_work, lwork_eig, d_info));
  } else {
    CLM_TRY(wwr::heevd<T>(cusolver_handle, wwr::WWRSOLVER_EIG_MODE_VECTOR, uplo, n, ws.U, n, ws.w,
                          ws.eig_work, lwork_eig, d_info));
  }

  // M = U diag(exp(w)) (fused kernel), then exp(A) = M U^H by one gemm -- U^T for
  // a real symmetric A, the conjugate transpose U^H for a complex Hermitian one.
  device::herm_exp_scale<T, RealT>(stream, n, ws.U, ws.w, ws.M);
  const wwr::wwrblasOperation_t op_uh =
      std::is_same_v<T, RealT> ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_C;
  CLM_TRY(wwr::gemm<T>(cublas_handle, wwr::WWRBLAS_OP_N, op_uh, n, n, n, &one, ws.M, n, ws.U, n,
                       &zero, d_expA, lde));

  return wwr::WWRBLAS_STATUS_SUCCESS;
}

// Instantiated once in instantiations.cpp -- the body names the .cu-side
// herm_exp_scale launcher declared only in the global module fragment, so an
// importer never re-instantiates it.
extern template Status expm_herm_bufferSize<float>(wwr::wwrsolverDnHandle_t, wwr::wwrblasFillMode_t,
                                                   int, std::size_t *);
extern template Status expm_herm_bufferSize<double>(wwr::wwrsolverDnHandle_t, wwr::wwrblasFillMode_t,
                                                    int, std::size_t *);
extern template Status
expm_herm_bufferSize<wwr::wwrFloatComplex>(wwr::wwrsolverDnHandle_t, wwr::wwrblasFillMode_t, int,
                                           std::size_t *);
extern template Status
expm_herm_bufferSize<wwr::wwrDoubleComplex>(wwr::wwrsolverDnHandle_t, wwr::wwrblasFillMode_t, int,
                                            std::size_t *);

extern template Status expm_herm<float>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                        wwr::wwrStream_t, wwr::wwrblasFillMode_t, int, const float *,
                                        int, float *, int, void *, std::size_t, int *);
extern template Status expm_herm<double>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t,
                                         wwr::wwrStream_t, wwr::wwrblasFillMode_t, int,
                                         const double *, int, double *, int, void *, std::size_t,
                                         int *);
extern template Status
expm_herm<wwr::wwrFloatComplex>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, wwr::wwrStream_t,
                                wwr::wwrblasFillMode_t, int, const wwr::wwrFloatComplex *, int,
                                wwr::wwrFloatComplex *, int, void *, std::size_t, int *);
extern template Status
expm_herm<wwr::wwrDoubleComplex>(wwr::wwrblasHandle_t, wwr::wwrsolverDnHandle_t, wwr::wwrStream_t,
                                 wwr::wwrblasFillMode_t, int, const wwr::wwrDoubleComplex *, int,
                                 wwr::wwrDoubleComplex *, int, void *, std::size_t, int *);

} // namespace calaman
