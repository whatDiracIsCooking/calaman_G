# Oracle suite for calaman.sterf -- eigenvalues of a symmetric tridiagonal. The
# oracle is LAPACKE_?sterf from calaman::lapack_reference; return early when
# absent so a missing oracle is a missing TIER named in the configure output,
# not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping sterf tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  sterf_tests
  SOURCES
  sterf_tests.cpp
  LINK
  calaman::sterf
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::extension::memory_buffer
)

# SterfOracleTests stages (d, e) on the device and runs the kernel, so it is
# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  sterf_tests
  SUITES
  SterfOracleTests
  REQUIRES_GPU
)

# SterfHostTests checks the n < 0 rejection, which launches nothing.
calaman_add_gtest_suite_tests(
  TARGET
  sterf_tests
  SUITES
  SterfHostTests
)
