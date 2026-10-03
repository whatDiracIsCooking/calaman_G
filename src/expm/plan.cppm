/**
 * @file plan.cppm
 * @brief Degree selection for the scaling-and-squaring ladder: the backward-error
 *        thresholds, the plan a given 1-norm implies, and what it costs
 *
 * The :plan partition of calaman.expm. Pure host arithmetic -- no device, no
 * BLAS -- so it is the honest answer to "what will this cost me?" ahead of the
 * call, and the one place that defines the degree/scaling policy expm() obeys.
 * pade_theta and expm_plan are left to implicit instantiation (no extern-template
 * list): they are trivial constexpr/host helpers.
 */

module;

#include "expm_bridge.h"

export module calaman.expm:plan;

import std;
import wwr.wrappers.common; // usual_fp, ComplexToRealType

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so each template carries its own
// `export`. pade_theta and expm_plan carry no such declaration (both are left to
// implicit instantiation), but the convention is kept for the whole partition.

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Backward-error thresholds
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// theta_m is the largest x with
//
//     sum_{k=2m+1}^{inf} |h_k| x^(k-1) <= u,     h(x) = log(exp(-x) r_m(x))
//
// Higham's backward-error criterion: r_m applied to the scaled matrix is the
// exact exponential of a nearby matrix A + E with norm(E) <= u * norm(A). The
// values come from evaluating that series in exact rational arithmetic and
// bisecting for the root at 60 significant digits (README has the derivation);
// at u = 2^-53 they reproduce Higham's published table exactly at m = 7, 9, 13.
//
//     m       u = 2^-53 (double)      u = 2^-24 (single)
//     3    1.495585217958292e-2    4.258730034897931e-1
//     5    2.539398330063232e-1    1.880152698533769
//     7    9.504178996162932e-1    3.925724846433284
//     9    2.097847961257067       6.249156334514102
//    13    5.371920351148152       1.124873763647540e1

/// @brief Backward-error threshold for degree @p m, or 0 if m is not on the ladder.
export template<wwr::usual_fp T>
constexpr wwr::ComplexToRealType<T> pade_theta(const int m) {
  using R = wwr::ComplexToRealType<T>;
  if constexpr (std::is_same_v<R, float>) {
    switch (m) {
    case 3:
      return 4.258730034897931e-1f;
    case 5:
      return 1.880152698533769f;
    case 7:
      return 3.925724846433284f;
    case 9:
      return 6.249156334514102f;
    case 13:
      return 1.124873763647540e1f;
    default:
      return R{0};
    }
  } else {
    switch (m) {
    case 3:
      return 1.495585217958292e-2;
    case 5:
      return 2.539398330063232e-1;
    case 7:
      return 9.504178996162932e-1;
    case 9:
      return 2.097847961257067;
    case 13:
      return 5.371920351148152;
    default:
      return R{0};
    }
  }
}

/// @brief The shape of one evaluation: which approximant, how many squarings, what it costs.
///
/// Both the prediction and the report: expm_plan() computes it from a norm ahead
/// of the call, and expm() fills it with the plan it executed. They are the same
/// type because expm is deterministic -- the degree and the scaling exponent
/// follow from the 1-norm alone, so what it WILL do and what it DID coincide, and
/// a second "info" struct would only restate a subset of this one.
export struct ExpmPlan {
  int m = 0;         ///< Pade degree chosen from {3, 5, 7, 9, 13}
  int s = 0;         ///< Squarings, i.e. the matrix is evaluated at A / 2^s
  int num_gemms = 0; ///< Matrix products the whole evaluation will issue
};

/**
 * @brief Pick the Pade degree and the scaling exponent for a given 1-norm.
 *
 * Higham's Algorithm 2.3: take the cheapest degree whose backward-error
 * threshold already covers the norm, and only fall back to scaling when even
 * theta_13 does not. Cost is pade_num_gemms(m) + s, and the ladder is ordered so
 * the first degree that fits is also the cheapest one that does. Exported
 * because it is the honest answer to "what will this cost me?" -- and the only
 * way a caller can predict the s that expm() reports.
 *
 * @param norm1 The matrix 1-norm; must be finite and non-negative.
 */
export template<wwr::usual_fp T>
ExpmPlan expm_plan(const wwr::ComplexToRealType<T> norm1) {
  using R = wwr::ComplexToRealType<T>;

  for (int i = 0; i < kNumPadeDegrees; ++i) {
    const int m = kPadeDegrees[i];
    if (norm1 <= pade_theta<T>(m)) {
      return ExpmPlan{m, 0, pade_num_gemms(m)};
    }
  }

  constexpr int m = 13;
  const R theta = pade_theta<T>(m);
  int s = static_cast<int>(std::ceil(std::log2(norm1 / theta)));
  if (s < 0) {
    s = 0;
  }
  return ExpmPlan{m, s, pade_num_gemms(m) + s};
}

} // namespace calaman
