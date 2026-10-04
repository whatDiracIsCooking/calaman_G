/**
 * @file validation.cppm
 * @brief The :validation partition of calaman.common -- small argument-checking
 *        predicates shared by every routine's front door
 *
 * A routine's first act is to reject bad arguments, and the same shapes recur
 * across src/: a fan of pointers that must all be non-null, a handful of
 * dimensions that must be ordered. This partition holds the reusable PREDICATES
 * for those checks -- pure, constexpr, backend-neutral, and testable on their
 * own -- so a front door composes them into one boolean rather than spelling the
 * fan out by hand. The early RETURN those predicates feed stays at the call site
 * (a function cannot return for its caller); CLM_REQUIRE in
 * error_handling/error_macros.h is the macro that pairs with them for it.
 *
 * Header-less, like :workspace: nothing device-side needs these, so they
 * live in the module purview only, reached through `import calaman.common;`.
 *
 * Usage:
 *   import calaman.common;
 *   if (!calaman::all_nonnull(d_a, d_b, d_c)) return bad_args_status;
 */

export module calaman.common:validation;

export namespace calaman {

/// @brief True iff every pointer passed is non-null
///
/// Folds `!= nullptr` over the whole pack, so one call replaces the
/// `a == nullptr || b == nullptr || ...` fan a front door would otherwise write.
/// The pointee types may differ and may be const; an empty pack is vacuously
/// true. Pure and noexcept -- it only compares addresses.
///
/// @tparam Ptrs Deduced pointee types (one per argument)
/// @param ptrs  The pointers to check
/// @return true when none of @p ptrs is null
template<typename... Ptrs>
[[nodiscard]] constexpr bool all_nonnull(const Ptrs *...ptrs) noexcept {
  return (... && (ptrs != nullptr));
}

} // namespace calaman
