# Oracle suite for calaman.tpttf -- packed to RFP triangular storage, checked
# bit-for-bit against LAPACKE_?tpttf. REQUIRES_GPU (excluded by
# `ctest -LE gpu`). Guard on calaman::lapack_reference and return early when it
# is absent -- a missing oracle is a missing tier, not a silent pass.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping tpttf tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  tpttf_tests
  SOURCES
  tpttf_tests.cpp
  LINK
  calaman::tpttf
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
  tpttf_tests
  SUITES
  TpttfOracleTests
  REQUIRES_GPU
)
