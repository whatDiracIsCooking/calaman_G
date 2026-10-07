# Oracle suite for calaman.lanht -- the ?lanht norm of a Hermitian tridiagonal,
# c/z. The oracle is the reference ?lanht, called through lapack.h (LAPACKE
# wraps no ?lanht), which arrives with calaman::lapack_reference; return early
# when absent so a missing oracle is a missing TIER named in the configure
# output, not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lanht tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lanht_tests
  SOURCES
  lanht_tests.cpp
  LINK
  calaman::lanht
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

# LanhtOracleTests stages (d, e) on the device -- even n == 0, a device memset
# -- so it is REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  lanht_tests
  SUITES
  LanhtOracleTests
  REQUIRES_GPU
)
