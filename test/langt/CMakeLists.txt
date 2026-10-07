# Oracle suite for calaman.langt -- the ?langt norm of a general tridiagonal,
# s/d/c/z. The oracle is the reference ?langt, called through lapack.h (LAPACKE
# wraps no ?langt), which arrives with calaman::lapack_reference; return early
# when absent so a missing oracle is a missing TIER named in the configure
# output, not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping langt tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  langt_tests
  SOURCES
  langt_tests.cpp
  LINK
  calaman::langt
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

# LangtOracleTests stages (dl, d, du) on the device -- even n == 0, a device
# memset -- so it is REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  langt_tests
  SUITES
  LangtOracleTests
  REQUIRES_GPU
)
