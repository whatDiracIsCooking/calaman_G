# Oracle suite for calaman.sytrs2 -- the level-3 symmetric Bunch-Kaufman solve.
# REQUIRES_GPU: every case stages A/ipiv/B/E on the device and runs the convert,
# trsm and solve kernels, so `ctest -LE gpu` excludes it. The oracle is
# LAPACKE_?sytrf (for the factor + pivots) plus LAPACKE_?sytrs (the forward
# solve; LAPACKE exposes no ?sytrs2) and a backward-error residual, all through
# calaman::lapack_reference; guard on that target and return early when absent,
# so a missing oracle is a missing TIER, not a silent pass (docs/architecture).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping sytrs2 tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  sytrs2_tests
  SOURCES
  sytrs2_tests.cpp
  LINK
  calaman.sytrs2
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
  sytrs2_tests
  SUITES
  Sytrs2OracleTests
  REQUIRES_GPU
)
