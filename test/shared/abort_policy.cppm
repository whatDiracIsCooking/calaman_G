/**
 * @file abort_policy.cppm
 * @brief Test-side abort-on-error policy
 *
 * WarpWraps ships no error policy: every gpu_check call and every RAII wrapper
 * names the policy it wants, because policy choice belongs to the consumer. The
 * suites here want plain abort-on-failure for the common "this must not fail"
 * case, so they own one rather than reaching into the library for a default that
 * does not exist.
 *
 * Deliberately under test/, not src/: a shipped calaman::AbortPolicy would make
 * this library decide error handling for ITS consumers -- the same coupling
 * WarpWraps removed by dropping its own. See test/shared/README.md.
 *
 * handle_error is noexcept so the policy is usable in the destruction slots
 * (nothrow_error_policy) as well as the create and device-access ones.
 *
 * Usage:
 *   import calaman.test.shared.abort_policy;
 *   using calaman::test::AbortPolicy;
 */

export module calaman.test.shared.abort_policy;

import wwr.extension.common; // success_code, error_name, error_string
import std;

export namespace calaman::test {

/**
 * @brief Error policy that prints to stderr and aborts
 *
 * @tparam T The error code type (any error_type: wwrError_t, a library status, ...)
 */
template<typename T>
class AbortPolicy {
public:
  using error_type = T;
  void handle_error(const T error, std::source_location location) noexcept {
    if (error != wwr::extension::success_code<T>()) {
      std::print(std::cerr, "GPU error at {}:{} in {}: {} ({})\n", location.file_name(),
                 location.line(), location.function_name(), wwr::extension::error_name(error),
                 wwr::extension::error_string(error));
      std::abort();
    }
  }
};

} // namespace calaman::test
