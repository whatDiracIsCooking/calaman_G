/**
 * @file cost_function.cppm
 * @brief What the CG algorithm on U(n) requires of a cost function
 *
 * Partition of calaman.cg_unitary.
 */

export module calaman.cg_unitary:cost_function;

import std;
import wwr.blas;          // wwrblasHandle_t, wwrblasStatus_t
import wwr.runtime_api;   // wwrStream_t
import wwr.wrappers.common; // usual_fp

export namespace calaman {

/// @brief Whether the solver descends or ascends.
///
/// The paper minimizes by default and maximizes via Remark 1 (§4), which flips
/// three signs and nothing else.
enum class CgDirection {
  Minimize,
  Maximize,
};

/// @brief -1 when minimizing, +1 when maximizing.
///
/// Multiplies the geodesic exponent and the derivative samples; see
/// geodesic_search for where each lands.
template<typename R>
constexpr R cg_sign(const CgDirection dir) {
  return (dir == CgDirection::Minimize) ? R(-1) : R(1);
}

/**
 * @brief The interface cg_unitary needs from a real-valued J on U(n).
 *
 * Four things, and nothing else:
 *
 * - `order`, the paper's q (§3.1): the highest degree at which t appears in the
 *   Taylor expansion of J(W + t Z) about t = 0. It fixes the highest frequency
 *   in the spectrum of the geodesic derivative, hence the search interval
 *   T_mu = 2 pi / (q |omega_max|) of eq. (15). It must be a compile-time
 *   constant and it must be right: too small and T_mu overshoots, breaking the
 *   at-most-one-cycle argument the line searches rest on. Brockett
 *   tr{W^H R W N} has q = 2; the JADE criterion has q = 4.
 *
 * - `euclidean_gradient`, writing Psi = dJ/dW(W). The Riemannian gradient of
 *   eq. (2) is assembled from it by the solver, not here.
 *
 * - `value`, writing J(W) as a *device* scalar. Device, because Table 2's
 *   step 13 compares N_DFT of them without a host round trip. The
 *   already-computed Psi is passed in so an implementation can reuse it -- every
 *   cost function here is a Frobenius inner product of W with something the
 *   gradient already formed -- and may be ignored by one that cannot. Only the
 *   real part is read.
 *
 * - `bufferSize`, the device workspace both of the above may use. The solver
 *   hands out a slice of its single caller-provided buffer; the functor owns the
 *   size, not the memory.
 */
template<typename F, typename T>
concept unitary_cost_function =
    wwr::usual_fp<T> &&
    requires(const F f, wwr::wwrblasHandle_t handle, wwr::wwrStream_t stream, int n, const T *d_W,
             int ldw, T *d_out, int ldo, const T *d_Psi, void *d_work, std::size_t lwork) {
      { F::order } -> std::convertible_to<int>;

      {
        f.euclidean_gradient(handle, stream, n, d_W, ldw, d_out, ldo, d_work, lwork)
      } -> std::same_as<wwr::wwrblasStatus_t>;

      {
        f.value(handle, stream, n, d_W, ldw, d_Psi, ldo, d_out, d_work, lwork)
      } -> std::same_as<wwr::wwrblasStatus_t>;

      { f.bufferSize(n) } -> std::same_as<std::size_t>;
    };

} // namespace calaman
