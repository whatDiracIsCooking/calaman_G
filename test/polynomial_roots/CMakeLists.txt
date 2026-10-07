# Oracle suite for calaman.polynomial_roots: polynomial roots against the
# eigenvalues of the companion matrix (LAPACKE_zgeev). Needs
# calaman::lapack_reference; a missing oracle is a missing tier named at
# configure, not a silent pass.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS
    "calaman::lapack_reference not found; skipping polynomial_roots tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  polynomial_roots_tests
  SOURCES
  polynomial_roots_tests.cpp
  LINK
  calaman::polynomial_roots
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  polynomial_roots_tests
  SUITES
  AberthOracleTests
  REQUIRES_GPU
)
