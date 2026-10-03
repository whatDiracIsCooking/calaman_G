/**
 * @file brockett.cppm
 * @brief The Brockett criterion on U(n): J(W) = trace{W^H R W N}
 *
 * Partition of calaman.cg_unitary.
 *
 * Paper §5.1. R is Hermitian and N is the diagonal matrix diag(1, ..., n).
 * *Maximizing* J drives W to the eigenvectors of R and W^H R W to a diagonal of
 * R's eigenvalues in ascending order -- so a Hermitian eigendecomposition is an
 * independent source of truth for what this solver should produce, which is what
 * makes it the right first cost function to carry.
 *
 * The Euclidean gradient is Psi = R W N, one gemm and one dgmm. The order is
 * q = 2: J(W + tZ) is exactly quadratic in t.
 */

export module calaman.cg_unitary:brockett;

import std;
import wwr.blas;            // wwrblasHandle_t, wwrblasStatus_t, WWRBLAS_*
import wwr.runtime_api;     // wwrStream_t
import wwr.complex;         // wwrFloatComplex, wwrDoubleComplex
import wwr.wrappers.common; // usual_fp, real_fp, ComplexToRealType
import wwr.wrappers.blas;   // gemm, dgmm, dot, dotc
import calaman.common;      // kOne, kZero

export namespace calaman {

/**
 * @brief Brockett cost function on U(n), as a cg_unitary cost functor.
 *
 * Holds references to its data; it does not own the device memory. Satisfies
 * unitary_cost_function<brockett_cost<T>, T>.
 *
 * @tparam T Element type (float, double, wwrFloatComplex, wwrDoubleComplex).
 */
template<wwr::usual_fp T>
struct brockett_cost {
  /// q for this cost function: trace{W^H R W N} is quadratic in W.
  static constexpr int order = 2;

  const T *d_R = nullptr; ///< n x n Hermitian, column-major.
  int ldr = 0;            ///< Leading dimension of d_R.
  const T *d_N = nullptr; ///< n diagonal entries, contiguous (typically 1..n).
  int dim = 0;            ///< n.

  /// One n x n block, for the R W product before the diagonal scaling.
  std::size_t bufferSize(const int n) const {
    return static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * sizeof(T);
  }

  /**
   * @brief Psi = R W N.
   *
   * dgmm applies diag(N) from the right, so N is stored as a vector of n entries
   * rather than an n x n matrix.
   */
  wwr::wwrblasStatus_t euclidean_gradient(wwr::wwrblasHandle_t handle, wwr::wwrStream_t /*stream*/,
                                          const int n, const T *d_W, const int ldw, T *d_Psi,
                                          const int ldp, void *d_work,
                                          const std::size_t lwork) const {
    if (lwork < bufferSize(n)) {
      return wwr::WWRBLAS_STATUS_ALLOC_FAILED;
    }
    T *d_tmp = static_cast<T *>(d_work);

    wwr::wwrblasPointerMode_t mode;
    wwr::wwrblasStatus_t status = wwr::wwrblasGetPointerMode(handle, &mode);
    if (status != wwr::WWRBLAS_STATUS_SUCCESS) {
      return status;
    }
    status = wwr::wwrblasSetPointerMode(handle, wwr::WWRBLAS_POINTER_MODE_HOST);
    if (status != wwr::WWRBLAS_STATUS_SUCCESS) {
      return status;
    }

    const auto restore = [&](const wwr::wwrblasStatus_t s) {
      wwr::wwrblasSetPointerMode(handle, mode);
      return s;
    };

    // tmp = R W
    status = wwr::gemm<T, int>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &kOne<T>, d_R,
                               ldr, d_W, ldw, &kZero<T>, d_tmp, n);
    if (status != wwr::WWRBLAS_STATUS_SUCCESS) {
      return restore(status);
    }

    // Psi = tmp diag(N)
    status = wwr::dgmm<T, int>(handle, wwr::WWRBLAS_SIDE_RIGHT, n, n, d_tmp, n, d_N, 1, d_Psi, ldp);
    return restore(status);
  }

  /**
   * @brief J(W) = Re trace{W^H (R W N)} = Re dotc(W, Psi), into a device scalar.
   *
   * The gradient is exactly the matrix the value needs, so when the caller
   * already has Psi this is a single level-1 reduction and no products at all.
   * Falls back to computing Psi when it is not supplied.
   *
   * @param d_Psi Optional; when null the gradient is computed into the workspace
   *              first, which then needs room for two n x n blocks.
   * @param d_out Device, one T. Only its real part is meaningful.
   */
  wwr::wwrblasStatus_t value(wwr::wwrblasHandle_t handle, wwr::wwrStream_t stream, const int n,
                             const T *d_W, const int ldw, const T *d_Psi, const int ldp, T *d_out,
                             void *d_work, const std::size_t lwork) const {
    const T *psi = d_Psi;
    const int ld_psi = ldp;

    if (psi == nullptr) {
      const std::size_t block = bufferSize(n);
      if (lwork < 2 * block) {
        return wwr::WWRBLAS_STATUS_ALLOC_FAILED;
      }
      T *d_own = static_cast<T *>(d_work);
      const wwr::wwrblasStatus_t s = euclidean_gradient(
          handle, stream, n, d_W, ldw, d_own, n, static_cast<std::byte *>(d_work) + block,
          lwork - block);
      if (s != wwr::WWRBLAS_STATUS_SUCCESS) {
        return s;
      }
      psi = d_own;
      return dot_into_device(handle, n, d_W, ldw, psi, n, d_out);
    }

    return dot_into_device(handle, n, d_W, ldw, psi, ld_psi, d_out);
  }

private:
  /// Re trace{W^H Psi} as a device scalar, via one dot in device pointer mode.
  /// Requires packed leading dimensions, which is how the solver allocates.
  static wwr::wwrblasStatus_t dot_into_device(wwr::wwrblasHandle_t handle, const int n,
                                              const T *d_W, const int ldw, const T *d_Psi,
                                              const int ldp, T *d_out) {
    if (ldw != n || ldp != n) {
      return wwr::WWRBLAS_STATUS_INVALID_VALUE;
    }

    wwr::wwrblasPointerMode_t mode;
    wwr::wwrblasStatus_t status = wwr::wwrblasGetPointerMode(handle, &mode);
    if (status != wwr::WWRBLAS_STATUS_SUCCESS) {
      return status;
    }
    status = wwr::wwrblasSetPointerMode(handle, wwr::WWRBLAS_POINTER_MODE_DEVICE);
    if (status != wwr::WWRBLAS_STATUS_SUCCESS) {
      return status;
    }

    if constexpr (wwr::real_fp<T>) {
      status = wwr::dot<T, int>(handle, n * n, d_W, 1, d_Psi, 1, d_out);
    } else {
      status = wwr::dotc<T, int>(handle, n * n, d_W, 1, d_Psi, 1, d_out);
    }

    wwr::wwrblasSetPointerMode(handle, mode);
    return status;
  }
};

} // namespace calaman
