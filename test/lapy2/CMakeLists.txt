# Oracle suite for calaman.lapy2 (built under src/lapack/lapy2/), and through it
# the shared lapy2_scalar in sym2x2.cuh. The oracle is the Fortran slapy2_ /
# dlapy2_ (no LAPACKE binding), so the suite NEEDS calaman::lapack_reference;
# return early when absent so the missing tier shows in the configure output.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lapy2 tests")
  return()
endif()

# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case runs the
# kernel on the device.
calaman_add_gtest_executable(
  NAME
  lapy2_tests
  SOURCES
  lapy2_tests.cpp
  LINK
  calaman::lapy2
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  lapy2_tests
  SUITES
  Lapy2OracleTests
  REQUIRES_GPU
)
