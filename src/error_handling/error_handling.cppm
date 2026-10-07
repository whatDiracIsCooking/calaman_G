/**
 * @file error_handling.cppm
 * @brief Primary interface for calaman.error_handling -- Status, the one return
 *        type the three WarpWraps error domains cast into
 *
 * WarpWraps reports failure in three separate, backend-neutral enums: a BLAS
 * call returns wwrblasStatus_t, a dense-solver call returns wwrsolverStatus_t,
 * and a runtime-API call (a device copy, an allocation, a stream sync) returns
 * wwrError_t. On either backend these are three DISTINCT vendor enums
 * (cublas/cusolver/cuda, or hipblas/hipsolver/hip), so a routine that touches
 * more than one domain has no single type to return. calaman.diff_norm hits
 * exactly this: its ell_inf path runs an iamax (BLAS) then reads one element
 * back with a copy (runtime), and today it surfaces a copy failure as
 * WWRBLAS_STATUS_NOT_INITIALIZED -- a BLAS code standing in for a runtime error,
 * because the return type is wwrblasStatus_t and there is nothing truer to say.
 * That is a lossy lie.
 *
 * Status is the one type all three cast INTO. It stores which @ref ErrorDomain
 * the code came from alongside the raw code, so a cross-domain routine returns a
 * single Status without discarding where the failure happened: the runtime copy
 * returns its wwrError_t AS a Status, and ok()/name()/message() still judge and
 * render it against the RIGHT domain's success constant and string tables.
 *
 * It is a VALUE TYPE, not an error POLICY -- and that is why it lives in src/
 * when AbortPolicy and DeviceHandle deliberately do not (test/shared/README.md).
 * A policy decides what to do on failure (abort, throw, log), and shipping one
 * forces that decision onto every consumer -- the coupling WarpWraps removed by
 * dropping its own. Status decides nothing: it carries the error and leaves the
 * response entirely to the caller, exactly as the bare wwr* enums do today. It
 * widens the vocabulary from "one domain's enum" to "any domain, tagged"; it
 * does not narrow the caller's choices.
 *
 * The three converting constructors are IMPLICIT on purpose: that is what lets a
 * routine write `return wwr::axpy(...);` or `return wwr::wwrMemcpy(...);` and
 * have either result become a Status with no ceremony at the call site.
 *
 * ok(), name() and message() forward to WarpWraps's own error_type traits
 * (success_code / error_name / error_string, specialised for all three types in
 * the wwr.extension.* layer), so there are no hand-copied success values or
 * string tables here to drift from the vendor headers. There is deliberately NO
 * operator bool: ok() is phrased positively (true == success), and a silent
 * bool conversion would collide with std::error_code's opposite convention
 * (true == error). Callers spell the check .ok().
 *
 * Status is itself a registered error_type: it specialises those same three
 * facilities for calaman::Status (at the foot of this file, in wwr::extension),
 * so wwr::extension::error_type<calaman::Status> holds and a Status plugs
 * straight into wwr::extension::gpu_check and any error_policy -- the same
 * machinery the wwr* enums use. gpu_check and a policy test `error ==
 * success_code<T>()`, so Status also defines operator== with OUTCOME semantics
 * (every success equals every other; errors compare by domain+code); see that
 * operator for why a single success sentinel has to match a success from any
 * domain.
 *
 * Usage:
 *   import calaman.error_handling;
 *   import wwr.blas;    // the status an axpy returns
 *   import std;
 *   calaman::Status s = wwr::axpy<float>(handle, n, &alpha, x, 1, y, 1);
 *   if (!s.ok()) {
 *     std::print(std::cerr, "axpy failed: {} ({})\n", s.name(), s.message());
 *   }
 *   // or, deferring the response to a policy, exactly as for a wwr* status:
 *   //   wwr::extension::gpu_check(s, calaman::test::AbortPolicy<calaman::Status>{});
 */

export module calaman.error_handling;

import wwr.runtime_api;      // wwrError_t (the runtime domain's type)
import wwr.extension.common; // success_code / error_name / error_string + the wwrError_t trait
import wwr.extension.blas;   // wwrblasStatus_t (re-exported) + its trait specialisation
import wwr.extension.solver; // wwrsolverStatus_t (re-exported) + its trait specialisation

// PointerModeStatus -- a recording error policy for ScopedPointerMode -- is a
// second error-handling value type that ships beside Status. `export import`
// re-exports it so a consumer (calaman.diff_norm) sees it through this one
// interface, exactly as calaman.common assembles its partitions.
export import :pointer_mode_policy;

export namespace calaman {

/// @brief Which WarpWraps error domain a Status was built from
///
/// Stamped by the constructor that ran, and read by ok()/name()/message() to
/// pick the matching success constant and string tables. One enumerator per
/// convertible type: adding a fourth domain means a constructor, an enumerator,
/// and a case in each of the three switches below.
enum class ErrorDomain { blas, solver, runtime };

/// @brief The one return type the three WarpWraps error domains cast into
///
/// Holds @ref domain (which enum @ref code came from) and @ref code (the vendor
/// enumerator, widened to int). Construct it by returning any of wwrblasStatus_t,
/// wwrsolverStatus_t or wwrError_t where a Status is expected -- the conversion
/// is implicit. Query success with ok() and render it with name()/message();
/// both resolve against @ref domain, so a runtime error never reads as a BLAS
/// one. The members are public: a caller that wants the raw code or domain back
/// (to switch on it, or re-cast it) has them directly.
struct Status {
  ErrorDomain domain; ///< Which domain @ref code belongs to
  int code;           ///< The vendor enumerator, widened to int

  /// @brief A BLAS status casts in, tagged ErrorDomain::blas
  constexpr Status(const wwr::wwrblasStatus_t status) noexcept
      : domain{ErrorDomain::blas}, code{static_cast<int>(status)} {}

  /// @brief A solver status casts in, tagged ErrorDomain::solver
  constexpr Status(const wwr::wwrsolverStatus_t status) noexcept
      : domain{ErrorDomain::solver}, code{static_cast<int>(status)} {}

  /// @brief A runtime-API error casts in, tagged ErrorDomain::runtime
  constexpr Status(const wwr::wwrError_t error) noexcept
      : domain{ErrorDomain::runtime}, code{static_cast<int>(error)} {}

  /// @brief True when @ref code is its own domain's success value
  ///
  /// Compares @ref code against wwr::extension::success_code<T>() for the T that
  /// @ref domain names, so each domain is judged by its OWN SUCCESS constant
  /// rather than a shared magic zero. Marked constexpr and constant-evaluable for
  /// the blas/solver domains (their success_code is constexpr upstream); the
  /// runtime domain's success_code is not, so a runtime Status's ok() evaluates
  /// only at run time.
  [[nodiscard]] constexpr bool ok() const noexcept {
    switch (domain) {
    case ErrorDomain::blas:
      return static_cast<wwr::wwrblasStatus_t>(code) ==
             wwr::extension::success_code<wwr::wwrblasStatus_t>();
    case ErrorDomain::solver:
      return static_cast<wwr::wwrsolverStatus_t>(code) ==
             wwr::extension::success_code<wwr::wwrsolverStatus_t>();
    case ErrorDomain::runtime:
      return static_cast<wwr::wwrError_t>(code) == wwr::extension::success_code<wwr::wwrError_t>();
    }
    return false; // unreachable: domain is always one of the three above
  }

  /// @brief The domain's short symbolic name for @ref code (the enumerator name)
  ///
  /// Forwards to wwr::extension::error_name for the T that @ref domain names.
  /// Never null -- an unrecognised code yields the domain's "unknown" spelling.
  [[nodiscard]] const char *name() const noexcept {
    switch (domain) {
    case ErrorDomain::blas:
      return wwr::extension::error_name(static_cast<wwr::wwrblasStatus_t>(code));
    case ErrorDomain::solver:
      return wwr::extension::error_name(static_cast<wwr::wwrsolverStatus_t>(code));
    case ErrorDomain::runtime:
      return wwr::extension::error_name(static_cast<wwr::wwrError_t>(code));
    }
    return ""; // unreachable: domain is always one of the three above
  }

  /// @brief The domain's human-readable description of @ref code
  ///
  /// Forwards to wwr::extension::error_string for the T that @ref domain names.
  /// Never null -- an unrecognised code yields the domain's "unknown" spelling.
  [[nodiscard]] const char *message() const noexcept {
    switch (domain) {
    case ErrorDomain::blas:
      return wwr::extension::error_string(static_cast<wwr::wwrblasStatus_t>(code));
    case ErrorDomain::solver:
      return wwr::extension::error_string(static_cast<wwr::wwrsolverStatus_t>(code));
    case ErrorDomain::runtime:
      return wwr::extension::error_string(static_cast<wwr::wwrError_t>(code));
    }
    return ""; // unreachable: domain is always one of the three above
  }

  /// @brief OUTCOME equality -- every success equals every other success
  ///
  /// A success has no meaningful domain to distinguish (there is no error to
  /// attribute), so all successful Statuses compare equal regardless of which
  /// domain reported them; two errors are equal only when they are the SAME code
  /// in the SAME domain. This is precisely what lets gpu_check and an error
  /// policy test `status == success_code<Status>()` and have it mean "did it
  /// succeed?" -- success_code<Status>() is one fixed value, yet it must match a
  /// success from any domain. (C++20 synthesises operator!= from this.) Note the
  /// deliberate consequence: two successes can be == while their @ref domain
  /// members differ -- the members stay queryable for diagnostics; equality asks
  /// only "same outcome?".
  [[nodiscard]] constexpr bool operator==(const Status &other) const noexcept {
    if (ok() || other.ok()) {
      return ok() && other.ok();
    }
    return domain == other.domain && code == other.code;
  }
};

/// @brief The host-side verdict on a device solver's devInfo: ok iff @p info is 0
///
/// The one place a nonzero devInfo becomes a Status. A failure carries the
/// BLAS-domain WWRBLAS_STATUS_INTERNAL_ERROR: the solver-domain status constants
/// are macros, so wwr.solver cannot export one to name here.
[[nodiscard]] constexpr Status devinfo_verdict(const int info) noexcept {
  return info == 0 ? Status{wwr::WWRBLAS_STATUS_SUCCESS}
                   : Status{wwr::WWRBLAS_STATUS_INTERNAL_ERROR};
}

} // namespace calaman

// The error_type registration for calaman::Status. These three explicit
// specializations of WarpWraps's facility templates (deleted primaries in
// wwr::extension) are what make `wwr::extension::error_type<calaman::Status>`
// hold, so a calaman routine returning Status plugs straight into gpu_check and
// any error_policy -- with Status::operator== above supplying the `== success`
// test both of those perform. They forward to Status's own members, so there is
// one implementation, not a parallel one. Declared in this module's purview (not
// the private fragment) so they are reachable to every importer; explicit
// specializations are not exportable and do not need to be.
namespace wwr::extension {

/// @brief The success value for calaman::Status (a success in any domain)
template<>
constexpr calaman::Status success_code<calaman::Status>() noexcept {
  return calaman::Status{wwr::WWRBLAS_STATUS_SUCCESS};
}

/// @brief The symbolic name of a Status's code, routed through its domain
template<>
inline const char *error_name<calaman::Status>(calaman::Status code) noexcept {
  return code.name();
}

/// @brief The human-readable description of a Status's code, via its domain
template<>
inline const char *error_string<calaman::Status>(calaman::Status code) noexcept {
  return code.message();
}

} // namespace wwr::extension
