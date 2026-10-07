# Oracle suite for calaman.lansf -- the ?lansf norm of a symmetric matrix in
# RFP storage, s/d. The oracle is the reference ?lansf (a Fortran symbol the
# test declares) plus LAPACKE_?trttf to build inputs, all arriving with
# calaman::lapack_reference; return early when absent so a missing oracle is a
# missing TIER named in the configure output, not a silent pass
# (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lansf tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lansf_tests
  SOURCES
  lansf_tests.cpp
  LINK
  calaman::lansf
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::extension::memory_buffer
)

# LansfOracleTests stages ARF on the device -- even n == 0, a device memset --
# so it is REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  lansf_tests
  SUITES
  LansfOracleTests
  REQUIRES_GPU
)
