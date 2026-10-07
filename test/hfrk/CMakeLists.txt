# Oracle suite for calaman.hfrk -- the rank-k update in RFP storage, ?sfrk
# (s/d) and ?hfrk (c/z). The oracle is LAPACKE_?sfrk / LAPACKE_?hfrk (plus
# LAPACKE_?tfttr to unpack results) from calaman::lapack_reference; return
# early when absent so a missing oracle is a missing TIER named in the
# configure output, not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping hfrk tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  hfrk_tests
  SOURCES
  hfrk_tests.cpp
  LINK
  calaman::hfrk
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

# Every case stages its operands on the device and runs herk/syrk and gemm, so
# it is REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  hfrk_tests
  SUITES
  HfrkOracleTests
  REQUIRES_GPU
)
