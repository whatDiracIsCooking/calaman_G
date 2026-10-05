# Oracle suite for calaman.hetrs -- the Hermitian Bunch-Kaufman solve.
# REQUIRES_GPU: every case stages A/ipiv/B on the device and runs the solve
# kernels, so `ctest -LE gpu` excludes it. The oracle is LAPACKE_?hetrf (for the
# factor + pivots) plus LAPACKE_?hetrs and a backward-error residual, through
# calaman::lapack_reference; guard on that target and return early when absent,
# so a missing oracle is a missing TIER, not a silent pass (docs/architecture).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping hetrs tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  hetrs_tests
  SOURCES
  hetrs_tests.cpp
  LINK
  calaman::hetrs
  calaman::common
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::test::shared::tolerance
  calaman::lapack_reference
  wwr::blas
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  hetrs_tests
  SUITES
  HetrsOracleTests
  REQUIRES_GPU
)
