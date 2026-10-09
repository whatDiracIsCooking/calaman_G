# Oracle suite for calaman.posv -- the positive definite solve driver.
# REQUIRES_GPU: every case factors (wwr::potrf) and solves (wwr::potrs) on the
# device, so `ctest -LE gpu` excludes it. The oracle is LAPACKE_?posv plus a
# backward-error residual, which arrives through calaman::lapack_reference;
# guard on that target and return early when it is absent -- a missing oracle is
# a missing TIER, not a silent pass (docs/architecture.md).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping posv tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  posv_tests
  SOURCES
  posv_tests.cpp
  LINK
  calaman::posv
  calaman::common
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::test::shared::tolerance
  calaman::lapack_reference
  wwr::blas
  wwr::solver
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  posv_tests
  SUITES
  PosvOracleTests
  REQUIRES_GPU
)
