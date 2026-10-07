# Oracle suite for calaman.lansb -- the ?lansb norm of a symmetric band matrix
# in LAPACK band storage, s/d/c/z. The oracle is the reference ?lansb
# (lapack.h's LAPACK_?lansb), arriving with calaman::lapack_reference; return
# early when absent so a missing oracle is a missing TIER named in the
# configure output, not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lansb tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lansb_tests
  SOURCES
  lansb_tests.cpp
  LINK
  calaman::lansb
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

# LansbOracleTests stages AB on the device -- even n == 0, a device memset --
# so it is REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  lansb_tests
  SUITES
  LansbOracleTests
  REQUIRES_GPU
)
