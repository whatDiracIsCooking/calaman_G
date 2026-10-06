# Oracle suite for calaman.tfttp -- RFP to packed triangular storage, checked
# bit-for-bit against LAPACKE_?tfttp. REQUIRES_GPU (excluded by
# `ctest -LE gpu`). Guard on calaman::lapack_reference and return early when it
# is absent -- a missing oracle is a missing tier, not a silent pass.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping tfttp tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  tfttp_tests
  SOURCES
  tfttp_tests.cpp
  LINK
  calaman::tfttp
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
  tfttp_tests
  SUITES
  TfttpOracleTests
  REQUIRES_GPU
)
