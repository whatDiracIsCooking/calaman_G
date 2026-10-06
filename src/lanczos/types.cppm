/**
 * @file types.cppm
 * @brief The Lanczos solver's options, result, end selection and matvec callback
 *
 * The :types partition of calaman.lanczos: the value types every other partition
 * names, kept apart from :solve so the Ritz-extraction stage can import them
 * without importing the driver.
 */

export module calaman.lanczos:types;

import std;
import wwr.runtime_api; // wwrStream_t
import calaman.common;  // real_fp
export import calaman.error_handling; // Status -- the callback's return type

export namespace calaman {

/// @brief Which end of the spectrum lanczos_solve converges.
enum class LanczosWhich {
  smallest,  ///< the nev algebraically smallest eigenvalues
  largest,   ///< the nev algebraically largest eigenvalues
  both_ends, ///< ceil(nev / 2) from the top, floor(nev / 2) from the bottom
};

/// @brief Convergence/restart knobs for one lanczos_solve call.
template<calaman::real_fp T>
struct LanczosOptions {
  /// @brief A Ritz pair is converged when its residual estimate |beta_m s_{m,i}|
  ///        is at or below tolerance * max(|theta_i|, ||T||).
  T tolerance = T{1e-8};
  /// @brief Thick restarts before giving up (0: a single cycle of ncv steps).
  int max_restarts = 100;
  /// @brief true: a non-converged solve returns WWRBLAS_STATUS_INTERNAL_ERROR
  ///        (result still filled); false: success, and the caller reads
  ///        LanczosResult::converged. As DavidsonOptions.
  bool fail_on_non_convergence = true;
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

/// @brief Outcome of a lanczos_solve call.
template<calaman::real_fp T>
struct LanczosResult {
  bool converged = false;
  int restarts = 0; ///< thick restarts performed
  int matvecs = 0;  ///< operator applications (LanczosMatvecFn calls)
  /// @brief The nev selected eigenvalues, ascending (host). Filled on every
  ///        completed run; accurate only when converged.
  std::vector<T> eigenvalues;
};

/// @brief matvec(stream, x, y): y = A x for one device vector of n elements.
///        A must be symmetric; @p x and @p y never alias.
template<calaman::real_fp T>
using LanczosMatvecFn = std::function<Status(wwr::wwrStream_t stream, const T *x, T *y)>;

} // namespace calaman
