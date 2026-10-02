/**
 * @file pointer_mode_policy.cppm
 * @brief The :pointer_mode_policy partition of calaman.error_handling --
 *        PointerModeStatus, a recording error policy for ScopedPointerMode
 *
 * wwr::extension::ScopedPointerMode takes an error_policy that decides what to do
 * when its get/set/restore of the handle's pointer mode fails. calaman ships no
 * abort or throw policy -- those live in test/shared on purpose (the handle/policy
 * boundary) -- yet a src/ routine that forces a pointer mode still needs those
 * failures to surface as its own return value, not a terminate() and not a
 * silently dropped error. PointerModeStatus is that middle path: it RECORDS the
 * first failing wwrblasStatus_t into a caller-owned slot and otherwise does
 * nothing, so the routine reads the slot after the guard leaves scope and folds it
 * into its Status. calaman.diff_norm is the first such caller.
 *
 * It lives beside Status because it is a second error-handling value type, not
 * device logic. Unlike the original in-module definition it is EXPORTED (via the
 * primary interface's `export import`), because a separate module -- diff_norm --
 * now names it across the module boundary; the test-only policies never cross into
 * src/, so they never needed exporting.
 */

export module calaman.error_handling:pointer_mode_policy;

import std;                // std::source_location (the error_policy contract's arg)
import wwr.extension.blas; // wwrblasStatus_t + WWRBLAS_STATUS_SUCCESS (re-exports wwr.blas)

export namespace calaman {

/// @brief An error policy for ScopedPointerMode that records the first BLAS failure
///
/// Models WarpWraps's error_policy: on each get/set/restore the guard calls
/// handle_error with that step's wwrblasStatus_t. This writes the FIRST non-success
/// status to @ref first_error (a caller-owned slot) and ignores the rest, so the
/// owning routine recovers the earliest failure as its Status. It does not abort,
/// throw or log -- the response is left entirely to the caller. handle_error is
/// noexcept because the guard's restore runs from a destructor, where a throw would
/// terminate; the defaulted source_location is unused but part of the contract.
struct PointerModeStatus {
  using error_type = wwr::wwrblasStatus_t;
  wwr::wwrblasStatus_t *first_error; ///< Caller-owned slot; the first failure lands here
  void handle_error(const wwr::wwrblasStatus_t status,
                    const std::source_location = std::source_location::current()) noexcept {
    if (status != wwr::WWRBLAS_STATUS_SUCCESS && *first_error == wwr::WWRBLAS_STATUS_SUCCESS) {
      *first_error = status;
    }
  }
};

} // namespace calaman
