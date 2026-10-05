# Oracle suite for calaman.hesv -- the Hermitian indefinite solve driver, which
# exercises the from-scratch factorization calaman.hetf2 end to end.
# REQUIRES_GPU: every case factors and solves on the device, so `ctest -LE gpu`
# excludes it. The oracle is LAPACKE_?hesv plus a backward-error residual,
# through calaman::lapack_reference; guard on that target and return early when
# absent, so a missing oracle is a missing TIER, not a silent pass
# (docs/architecture).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping hesv tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  hesv_tests
  SOURCES
  hesv_tests.cpp
  LINK
  calaman::hesv
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
  hesv_tests
  SUITES
  HesvOracleTests
  REQUIRES_GPU
)
