# Oracle suite for calaman.lantr -- the ?lantr norm of a triangular
# (trapezoidal) matrix, s/d/c/z. The oracle is the reference ?lantr (lapack.h's
# LAPACK_?lantr), arriving with calaman::lapack_reference; return early when
# absent so a missing oracle is a missing TIER named in the configure output,
# not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lantr tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lantr_tests
  SOURCES
  lantr_tests.cpp
  LINK
  calaman::lantr
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

# LantrOracleTests stages A on the device -- even m == 0, a device memset --
# so it is REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  lantr_tests
  SUITES
  LantrOracleTests
  REQUIRES_GPU
)
