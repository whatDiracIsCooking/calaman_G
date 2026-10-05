# Oracle test for calaman.larft (forming a block reflector's triangular factor).
# REQUIRES_GPU: every case stages V, tau and T on the device and runs the
# trmm/gemm/geam kernels, so `ctest -LE gpu` excludes it. It compares against
# LAPACKE_?larft, which arrives as calaman::lapack_reference; guard on
# that target and return early when it is absent -- a missing oracle is a
# missing tier named in the configure output, not a silent pass (see
# docs/architecture.md section 3). The two test/shared/ modules plus
# wwr.extension.memory_buffer stage the device buffers; larft allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping larft tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  larft_tests
  SOURCES
  larft_tests.cpp
  LINK
  calaman::larft
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::test::shared::tolerance
  calaman::lapack_reference
  wwr::blas
  wwr::runtime_api
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  larft_tests
  SUITES
  LarftOracleTests
  REQUIRES_GPU
)
