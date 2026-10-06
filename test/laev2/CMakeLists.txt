# Oracle suite for calaman.laev2 (built under src/lapack/laev2/), and through it
# the shared laev2_scalar in sym2x2.cuh. The oracle is the Fortran slaev2_ /
# dlaev2_ (no LAPACKE binding), so the suite NEEDS calaman::lapack_reference;
# return early when absent so the missing tier shows in the configure output.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping laev2 tests")
  return()
endif()

# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case runs the
# kernel on the device.
calaman_add_gtest_executable(
  NAME
  laev2_tests
  SOURCES
  laev2_tests.cpp
  LINK
  calaman::laev2
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
  laev2_tests
  SUITES
  Laev2OracleTests
  REQUIRES_GPU
)
