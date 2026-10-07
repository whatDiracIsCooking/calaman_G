# Oracle suite for calaman.lacn2 (built under src/lapack/lacn2/) -- the 1-norm
# estimator. The oracle is the reference ?lacn2, declared by Fortran symbol in
# the suite (LAPACKE does not wrap it), plus LAPACKE_?lange for the exact norm;
# both arrive with calaman::lapack_reference, so return early when it is absent
# -- a missing oracle is a missing TIER, not a silent pass.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lacn2 tests")
  return()
endif()

# REQUIRES_GPU (labeled `gpu`): the device loop answers each kase with
# wwr::gemv on the card.
calaman_add_gtest_executable(
  NAME
  lacn2_tests
  SOURCES
  lacn2_tests.cpp
  LINK
  calaman::lacn2
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::blas
  wwr::wrappers::blas
  wwr::runtime_api
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  lacn2_tests
  SUITES
  Lacn2OracleTests
  REQUIRES_GPU
)
