/**
 * @file error_macros.h
 * @brief CLM_TRY -- evaluate a WarpWraps call, cast its result to Status, and
 *        early-return it unless it succeeded
 *
 * A plain header, not a module unit, because this is a MACRO and macros do not
 * cross a module boundary: `import calaman.error_handling;` would not carry one.
 * So the early-return check that every Status-returning routine repeats --
 *
 *   const auto status = wwr::axpy<T>(...);
 *   if (status != wwr::WWRBLAS_STATUS_SUCCESS) { return status; }
 *
 * -- collapses to `CLM_TRY(wwr::axpy<T>(...));` only by #include, pulled into a
 * consumer's global module fragment the same way "common/enums.h" is. This is
 * the shared, cross-domain generalisation of gebal's local CLM_GEBAL_CHECK:
 * that macro is hard-wired to one domain (wwrError_t == wwrSuccess), while
 * CLM_TRY routes through @ref calaman::Status, so a blas, solver OR runtime
 * result is judged by its OWN domain's success constant via Status::ok().
 *
 * Two requirements at the expansion site, both satisfied wherever a routine
 * already returns a Status:
 *   - calaman::Status must be visible -- `import calaman.error_handling;`. The
 *     macro spells it ::calaman::Status fully, so nothing need be `using`-ed.
 *   - the enclosing function must return Status (or something Status converts
 *     to); CLM_TRY's bail-out is `return <the Status>`. A routine that returns a
 *     bare wwr* enum instead cannot use CLM_TRY -- it is for the Status
 *     convention, not a drop-in for CLM_GEBAL_CHECK.
 *
 * The cast is the point: @p expr is a wwr* enum (wwrblasStatus_t /
 * wwrsolverStatus_t / wwrError_t), and copy-initialising a Status from it fires
 * the matching implicit converting constructor, tagging the right ErrorDomain.
 * ok() then compares against that domain's success value -- so CLM_TRY is also
 * MORE correct than the hand-written check it replaces, which tests every
 * domain's result against the single BLAS success constant.
 *
 * Consumers #include this by its root-relative path, "error_handling/error_macros.h".
 */

#pragma once

/// @brief Early-return a WarpWraps result as a Status unless it succeeded
///
/// Evaluates @p expr once, casts the result to ::calaman::Status (the implicit
/// converting constructor tags its ErrorDomain), and -- if it is not ok() --
/// returns that Status from the enclosing function. Wrapped in do/while(0) so it
/// is one statement and the temporary does not leak; the identifier is
/// uglified to avoid colliding with names in @p expr.
///
/// @param expr A WarpWraps call returning wwrblasStatus_t, wwrsolverStatus_t or
///             wwrError_t. Evaluated exactly once.
#define CLM_TRY(expr)                                                                               \
  do {                                                                                              \
    const ::calaman::Status clm_try_status_ = (expr);                                               \
    if (!clm_try_status_.ok()) {                                                                    \
      return clm_try_status_;                                                                       \
    }                                                                                               \
  } while (0)
