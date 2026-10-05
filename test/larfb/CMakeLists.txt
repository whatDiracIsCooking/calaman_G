# Oracle test for calaman.larfb (applying a block reflector to a matrix).
# REQUIRES_GPU: every case stages V, T, C and the workspace on the device and
# runs the trmm/gemm/geam kernels, so `ctest -LE gpu` excludes it. It compares
# against LAPACKE_?larfb, which arrives as calaman::lapack_reference; guard on
# that target and return early when it is absent -- a missing oracle is a
# missing tier named in the configure output, not a silent pass (see
# docs/architecture.md section 3). The two test/shared/ modules plus
# wwr.extension.memory_buffer stage the device buffers; larfb allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping larfb tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  larfb_tests
  SOURCES
  larfb_tests.cpp
  LINK
  calaman::larfb
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
  larfb_tests
  SUITES
  LarfbOracleTests
  REQUIRES_GPU
)
