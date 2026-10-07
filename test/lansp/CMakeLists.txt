# Oracle suite for calaman.lansp -- the ?lansp norm of a symmetric matrix in
# packed storage, s/d/c/z. The oracle is the reference ?lansp (lapack.h's
# LAPACK_?lansp) plus LAPACKE_?trttp to build inputs, all arriving with
# calaman::lapack_reference; return early when absent so a missing oracle is a
# missing TIER named in the configure output, not a silent pass
# (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lansp tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lansp_tests
  SOURCES
  lansp_tests.cpp
  LINK
  calaman::lansp
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

# LanspOracleTests stages AP on the device -- even n == 0, a device memset --
# so it is REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  lansp_tests
  SUITES
  LanspOracleTests
  REQUIRES_GPU
)
