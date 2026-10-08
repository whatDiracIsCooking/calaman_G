/**
 * @file types.cppm
 * @brief The Lanczos solver's options, info, end selection, and the matvec
 *        callback with its linear_operator adapter
 *
 * The :types partition of calaman.lanczos: the value types every other partition
 * names, kept apart from :solve so the Ritz-extraction stage can import them
 * without importing the driver.
 */

export module calaman.lanczos:types;

import std;
import wwr.blas;        // WWRBLAS_STATUS_SUCCESS
import wwr.runtime_api; // wwrStream_t
import calaman.common;  // real_fp
export import calaman.error_handling;  // Status -- the callback's return type
export import calaman.iterative;       // IterationInfo, stop_reason, converged
export import calaman.linear_operator; // linear_operator -- what the driver applies
export import calaman.ritz;            // RitzWhich (LanczosWhich), RitzSelection

export namespace calaman {

/// @brief Which end of the spectrum lanczos_solve converges (count = nev).
using LanczosWhich = RitzWhich;

/// @brief Convergence/restart knobs for one lanczos_solve call.
template<calaman::real_fp T>
struct LanczosOptions {
  /// @brief Relative: a pair converges when |beta_m s_{m,i}| (and, verified,
  ///        ||A x - theta x||_2) <= tol * max(|theta|, ||T||_2), ||T||_2 = max |theta|
  ///        over the subspace spectrum (classify_ritz). Compared: architecture.md §8.
  ///        Shift-invert: bounds theta of (A - sigma I)^{-1}, not lambda (§9).
  T tolerance = T{1e-8};
  /// @brief Thick restarts before giving up (0: a single cycle of ncv steps).
  int max_iterations = 100;
  /// @brief true: once the estimates pass, also require each true residual
  ///        ||A x_i - theta_i x_i||_2 under the same bound (nev more matvecs per
  ///        check); a miss keeps restarting.
  bool verify_residuals = false;
  /// @brief Seed for the generator states behind the random start vector and
  ///        breakdown-recovery vectors.
  std::uint64_t seed = 0x5eedULL;
  /// @brief OPTIONAL start vector: n elements on the device, need not be
  ///        normalised. Null (default) draws a random normal one from @c seed.
  const T *start_vector = nullptr;
};

/// @brief Why lanczos_solve stopped. A breakdown is recovered, not a stop.
enum class LanczosStopReason {
  Converged,        ///< every wanted pair's estimate (and true residual, if verified) passed
  MaxIterations,    ///< max_iterations thick restarts ran out
  NumericalFailure, ///< a BLAS/solver/runtime call or the matvec failed (Status says which)
};
static_assert(stop_reason<LanczosStopReason>);

/// @brief What lanczos_solve did: iterations are thick restarts performed.
struct LanczosInfo : IterationInfo<LanczosStopReason> {
  int matvecs = 0; ///< operator applications (matvec calls)
};

/// @brief matvec(stream, x, y): y = A x for one device vector of n elements.
///        A must be symmetric; @p x and @p y never alias.
template<typename F, typename T>
concept lanczos_matvec = requires(const F &f, wwr::wwrStream_t stream, const T *x, T *y) {
  { f(stream, x, y) } -> std::convertible_to<Status>;
};

/// @brief The linear_operator over a lanczos_matvec: apply calls @p matvec once
///        per column (lanczos_solve applies k = 1). Holds a reference to it.
template<calaman::real_fp T, lanczos_matvec<T> F>
struct LanczosMatvecOperator {
  const F &matvec;
  int n; ///< the vector length: the stride between columns of X and of Y

  Status apply(wwr::wwrStream_t stream, const int k, const T *X, T *Y) {
    const auto nz = static_cast<std::size_t>(n);
    for (std::size_t j = 0; j < static_cast<std::size_t>(k); ++j) {
      const Status status = matvec(stream, X + j * nz, Y + j * nz);
      if (!status.ok()) {
        return status;
      }
    }
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
};

} // namespace calaman
