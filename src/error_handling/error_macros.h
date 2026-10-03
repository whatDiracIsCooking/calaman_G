/**
 * @file error_macros.h
 * @brief The two early-return guards a routine's body repeats -- CLM_TRY for a
 *        call that may fail, CLM_REQUIRE for a precondition that must hold
 *
 * A plain header, not a module unit, because this is a MACRO and macros do not
 * cross a module boundary: `import calaman.error_handling;` would not carry one.
 * So the early-return check that every Status-returning routine repeats --
 *
 *   const auto status = wwr::axpy<T>(...);
 *   if (status != wwr::WWRBLAS_STATUS_SUCCESS) { return status; }
 *
 * -- collapses to `CLM_TRY(wwr::axpy<T>(...));` only by #include, pulled into a
 * consumer's global module fragment the same way "common/enums.h" is. This
 * generalised, and has now replaced, gebal's former local CLM_GEBAL_CHECK:
 * that macro was hard-wired to one domain (wwrError_t == wwrSuccess), while
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
 *     convention, and such a routine must spell the check by hand.
 *
 * The cast is the point: @p expr is a wwr* enum (wwrblasStatus_t /
 * wwrsolverStatus_t / wwrError_t), and copy-initialising a Status from it fires
 * the matching implicit converting constructor, tagging the right ErrorDomain.
 * ok() then compares against that domain's success value -- so CLM_TRY is also
 * MORE correct than the hand-written check it replaces, which tests every
 * domain's result against the single BLAS success constant.
 *
 * CLM_REQUIRE is the FRONT-DOOR counterpart: where CLM_TRY propagates a failed
 * call, CLM_REQUIRE guards a precondition. It is the generalisation of the
 * `if (bad args) { return <sentinel>; }` block every routine opens with -- the
 * argument validation in calaman.horner / calaman.expm and the rest. Unlike
 * CLM_TRY it touches no Status and needs nothing imported: it simply returns the
 * sentinel the caller names, so it works whether the enclosing function returns
 * a bare wwr* enum or a calaman::Status (the enum converts). It pairs with the
 * predicate helpers in calaman.common's :validation partition (all_nonnull),
 * which do the checking a function CAN express, leaving the macro only the early
 * return it cannot.
 *
 * Consumers #include this by its root-relative path, "error_handling/error_macros.h".
 */

#pragma once

/// @brief Early-return a WarpWraps result as a Status unless it succeeded
///
/// Evaluates the expression once, casts the result to ::calaman::Status (the
/// implicit converting constructor tags its ErrorDomain), and -- if it is not
/// ok() -- returns that Status from the enclosing function. Wrapped in
/// do/while(0) so it is one statement and the temporary does not leak; the
/// identifier is uglified to avoid colliding with names in the expression.
///
/// Variadic on purpose: the whole expression is one logical argument, but a call
/// like `CLM_TRY(wwr::iamax<T, int>(...))` carries an unparenthesised comma in
/// its template-argument list, which the preprocessor would otherwise read as an
/// argument separator. Taking `...` and re-joining with __VA_ARGS__ lets such a
/// comma pass through untouched, so a two-template-argument WarpWraps call needs
/// no defensive extra parentheses at the call site.
///
/// @param ... A WarpWraps call returning wwrblasStatus_t, wwrsolverStatus_t or
///            wwrError_t. Evaluated exactly once.
#define CLM_TRY(...)                                                                                \
  do {                                                                                              \
    const ::calaman::Status clm_try_status_ = (__VA_ARGS__);                                        \
    if (!clm_try_status_.ok()) {                                                                    \
      return clm_try_status_;                                                                       \
    }                                                                                               \
  } while (0)

/// @brief Early-return @p status from the enclosing function unless @p cond holds
///
/// The front-door guard: @p cond is the precondition that must HOLD (write the
/// invariant, not the failure case, so it reads like the @pre it enforces), and
/// if it is false the macro returns @p status. @p status is evaluated only on the
/// failing path and may be anything the enclosing function's return type accepts
/// -- a wwr* enum, a ::calaman::Status, a std::size_t sentinel. Wrapped in
/// do/while(0) so it is one statement. Needs nothing imported.
///
/// @param cond   The precondition; the guard fires (returns) when it is false.
/// @param status The value to return when @p cond does not hold.
#define CLM_REQUIRE(cond, status)                                                                   \
  do {                                                                                              \
    if (!(cond)) {                                                                                  \
      return (status);                                                                              \
    }                                                                                               \
  } while (0)
