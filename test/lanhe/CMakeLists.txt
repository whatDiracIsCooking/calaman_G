# Oracle suite for calaman.lanhe -- the ?lanhe norm of a Hermitian matrix from
# one triangle, c/z. The oracle is the reference ?lanhe through LAPACKE, which
# arrives with calaman::lapack_reference; return early when absent so a missing
# oracle is a missing TIER named in the configure output, not a silent pass
# (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lanhe tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lanhe_tests
  SOURCES
  lanhe_tests.cpp
  LINK
  calaman::lanhe
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

# LanheOracleTests stages A on the device -- even n == 0, a device memset -- so
# it is REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  lanhe_tests
  SUITES
  LanheOracleTests
  REQUIRES_GPU
)
