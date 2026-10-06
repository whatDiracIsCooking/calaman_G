# Oracle suite for calaman.tfttr -- RFP to full triangular storage, checked
# bit-for-bit against LAPACKE_?tfttr. REQUIRES_GPU (excluded by
# `ctest -LE gpu`). Guard on calaman::lapack_reference and return early when it
# is absent -- a missing oracle is a missing tier, not a silent pass.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping tfttr tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  tfttr_tests
  SOURCES
  tfttr_tests.cpp
  LINK
  calaman::tfttr
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
  tfttr_tests
  SUITES
  TfttrOracleTests
  REQUIRES_GPU
)
