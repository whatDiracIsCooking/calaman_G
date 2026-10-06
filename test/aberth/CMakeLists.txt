# Oracle suite for calaman.aberth: polynomial roots against the eigenvalues of
# the companion matrix (LAPACKE_zgeev). Needs calaman::lapack_reference; a
# missing oracle is a missing tier named at configure, not a silent pass.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping aberth tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  aberth_tests
  SOURCES
  aberth_tests.cpp
  LINK
  calaman::aberth
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  aberth_tests
  SUITES
  AberthOracleTests
  REQUIRES_GPU
)
