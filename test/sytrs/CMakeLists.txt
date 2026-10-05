# Oracle suite for calaman.sytrs -- the symmetric Bunch-Kaufman solve.
# REQUIRES_GPU: every case stages A/ipiv/B on the device and runs the kernels,
# so `ctest -LE gpu` excludes it. The oracle is LAPACKE_?sytrf (to produce the
# factor + pivots) plus a backward-error residual, which arrives through
# calaman::lapack_reference; guard on that target and return early when it is
# absent -- a missing oracle is a missing TIER, not a silent pass
# (docs/architecture.md). sytrs takes a bare wwrblasHandle_t plus caller device
# arrays and allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping sytrs tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  sytrs_tests
  SOURCES
  sytrs_tests.cpp
  LINK
  calaman.sytrs
  calaman.common
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.utils.shared_device
  calaman.test.shared.tolerance
  calaman::lapack_reference
  wwr.blas
  wwr.runtime_api
  wwr.complex
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  sytrs_tests
  SUITES
  SytrsOracleTests
  REQUIRES_GPU
)
