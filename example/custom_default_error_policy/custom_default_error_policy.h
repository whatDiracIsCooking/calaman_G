// custom_default_error_policy.h -- a custom DEFAULT error policy for wwr.
//
// Point the CMake cache variable WWR_DEFAULT_ERROR_POLICY_IMPL at this file and
// wwr::extension::DefaultErrorPolicy<T> derives from the type below instead of
// carrying its built-in "print to stderr and std::abort()" body. The swap is
// build-wide: EVERY default-policy slot in the extension layer picks it up --
// the argless gpu_check, and the P_create / P_alloc / P_free / P_destroy
// template-argument defaults on the RAII buffer, handle, stream and plan
// wrappers -- with no change at any of those call sites.
//
// This is an ordinary header, #included into the error_handling module's global
// module fragment (see src/extension/common/error_handling/default_error_policy.cppm).
// So it brings its own #includes and must not `import` anything.
//
// CONTRACT -- handle_error MUST be noexcept. The destruction-slot policies
// (P_free / P_destroy) are constrained by nothrow_error_policy: they run from
// destructors, where a throwing handler would std::terminate. A custom default
// that threw would therefore fail to compile those slots. This one logs and
// returns instead of throwing or aborting -- a "record the failure, don't
// crash" policy, the opposite end from the built-in default.

#ifndef WWR_EXAMPLE_CUSTOM_DEFAULT_ERROR_POLICY_H
#define WWR_EXAMPLE_CUSTOM_DEFAULT_ERROR_POLICY_H

#include <iostream>
#include <print>
#include <source_location>
#include <utility>

namespace wwr::extension {

// The customization point wwr looks for when WWR_DEFAULT_ERROR_POLICY_IMPL is
// set. Generic over the error code type T (gpuError_t and each library status
// type), exactly like the built-in default it replaces.
template <typename T>
struct DefaultErrorPolicyImpl {
  using error_type = T;

  void handle_error(T error, std::source_location location) noexcept {
    std::println(std::cerr, "[custom policy] GPU error {} at {}:{} in {} -- continuing (no abort)",
                 std::to_underlying(error), location.file_name(), location.line(),
                 location.function_name());
  }
};

} // namespace wwr::extension

#endif // WWR_EXAMPLE_CUSTOM_DEFAULT_ERROR_POLICY_H
