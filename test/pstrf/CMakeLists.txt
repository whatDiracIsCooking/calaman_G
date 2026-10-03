# Oracle suite for calaman.pstrf -- the blocked (level-3) pivoted Cholesky driver.
# REQUIRES_GPU: every case stages A/work on the device and runs the pivot and
# factor kernels, so `ctest -LE gpu` excludes it. The oracle is LAPACKE_?pstrf
# over the whole matrix, which arrives as calaman::lapack_reference; guard on that
# target and return early when it is absent -- a missing oracle is a missing TIER
# named in the configure output, not a silent pass (docs/architecture.md). The two
# test/shared/ modules stage the device buffers and tolerance supplies the
# backward-error bound. pstrf takes a bare wwrblasHandle_t plus caller device
# arrays and allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping pstrf tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  pstrf_tests
  SOURCES
  pstrf_tests.cpp
  LINK
  calaman.pstrf
  calaman.common
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.utils.shared_device
  calaman.test.shared.tolerance
  calaman::lapack_reference
  wwr.blas
  wwr.runtime_api
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  pstrf_tests
  SUITES
  PstrfOracleTests
  REQUIRES_GPU
)
