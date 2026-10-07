# Oracle suite for calaman.lanhf -- the ?lanhf norm of a Hermitian matrix in
# RFP storage, c/z. The oracle is the reference ?lanhf (a Fortran symbol the
# test declares) plus LAPACKE_?trttf to build inputs, all arriving with
# calaman::lapack_reference; return early when absent so a missing oracle is a
# missing TIER named in the configure output, not a silent pass
# (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lanhf tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lanhf_tests
  SOURCES
  lanhf_tests.cpp
  LINK
  calaman::lanhf
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

# LanhfOracleTests stages ARF on the device -- even n == 0, a device memset --
# so it is REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  lanhf_tests
  SUITES
  LanhfOracleTests
  REQUIRES_GPU
)
