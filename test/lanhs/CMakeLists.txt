# Oracle suite for calaman.lanhs -- the ?lanhs norm of an upper Hessenberg
# matrix, s/d/c/z. The oracle is the reference ?lanhs through lapack.h (which
# lapacke.h includes; LAPACKE has no ?lanhs wrapper), arriving with
# calaman::lapack_reference; return early when absent so a missing oracle is a
# missing TIER named in the configure output, not a silent pass
# (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lanhs tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lanhs_tests
  SOURCES
  lanhs_tests.cpp
  LINK
  calaman::lanhs
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

# LanhsOracleTests stages A on the device -- even n == 0, a device memset -- so
# it is REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  lanhs_tests
  SUITES
  LanhsOracleTests
  REQUIRES_GPU
)
