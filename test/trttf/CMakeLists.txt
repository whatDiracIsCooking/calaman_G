# Oracle suite for calaman.trttf -- full to RFP triangular storage, checked
# bit-for-bit against LAPACKE_?trttf. REQUIRES_GPU (excluded by
# `ctest -LE gpu`). Guard on calaman::lapack_reference and return early when it
# is absent -- a missing oracle is a missing tier, not a silent pass.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping trttf tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  trttf_tests
  SOURCES
  trttf_tests.cpp
  LINK
  calaman::trttf
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
  trttf_tests
  SUITES
  TrttfOracleTests
  REQUIRES_GPU
)
