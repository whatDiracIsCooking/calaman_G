# Oracle suite for calaman.tfsm -- the triangular solve with A in RFP storage,
# ?tfsm for s/d/c/z. The oracle is LAPACKE_?tfsm (plus LAPACKE_?trttf to pack
# the inputs) from calaman::lapack_reference; return early when absent so a
# missing oracle is a missing TIER named in the configure output, not a silent
# pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping tfsm tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  tfsm_tests
  SOURCES
  tfsm_tests.cpp
  LINK
  calaman::tfsm
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::blas
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

# Every case stages its operands on the device and runs trsm and gemm, so it
# is REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  tfsm_tests
  SUITES
  TfsmOracleTests
  REQUIRES_GPU
)
