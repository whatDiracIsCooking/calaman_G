/**
 * @file iterative.cppm
 * @brief The calaman.iterative module -- the outcome report every iterative
 *        method shares: IterationInfo, the stop_reason concept, converged()
 *
 * Each iterative method keeps its own stop-reason enum. stop_reason requires
 * the three every method has -- Converged, MaxIterations, NumericalFailure --
 * and a method may add its own (cg's LineSearchFailed, feast's
 * SubspaceTooSmall). There is no shared union enum. A method's Info struct
 * derives from IterationInfo<ItsReason> and adds its own convergence measure:
 * only fields that mean the same thing in every method live here.
 *
 * Non-convergence is an outcome, not an error. An iterative routine returns
 * success whenever it stops on one of its own conditions, MaxIterations
 * included; the caller reads converged(info) or info.reason. A failing Status
 * means a fault: a bad argument, or a BLAS/solver/runtime call that failed.
 *
 * The iteration budget is spelled max_iterations in every method's Options, to
 * match IterationInfo::iterations.
 *
 * Imports only std: no device code, no WarpWraps.
 */

export module calaman.iterative;

import std;

export namespace calaman {

/// @brief An enum with at least the three stop reasons every iterative method shares.
template<class E>
concept stop_reason = std::is_enum_v<E> && requires {
  E::Converged;
  E::MaxIterations;
  E::NumericalFailure;
};

/// @brief What an iterative method did: the base of each method's Info struct.
template<stop_reason Reason>
struct IterationInfo {
  int iterations = 0; ///< outer iterations completed, in the method's own unit
  Reason reason = Reason::MaxIterations;
};

/// @brief True when the method stopped because it converged.
template<stop_reason Reason>
constexpr bool converged(const IterationInfo<Reason> &i) noexcept {
  return i.reason == Reason::Converged;
}

} // namespace calaman
