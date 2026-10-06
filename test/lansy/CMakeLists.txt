# Oracle suite for calaman.lansy -- the ?lansy norm of a symmetric matrix from
# one triangle. The oracle is the reference ?lansy through LAPACKE, which
# arrives with calaman::lapack_reference; return early when absent so a missing
# oracle is a missing TIER named in the configure output, not a silent pass
# (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lansy tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lansy_tests
  SOURCES
  lansy_tests.cpp
  LINK
  calaman::lansy
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::extension::memory_buffer
)

# LansyOracleTests stages A on the device -- even n == 0, a device memset -- so
# it is REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  lansy_tests
  SUITES
  LansyOracleTests
  REQUIRES_GPU
)
