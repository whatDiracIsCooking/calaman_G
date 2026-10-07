# Oracle suite for calaman.lapy3 (built under src/lapack/lapy3/), and through it
# the header-only lapy3_scalar in lapy3.cuh. The oracle is the Fortran slapy3_ /
# dlapy3_ (no LAPACKE binding), so the suite NEEDS calaman::lapack_reference;
# return early when absent so the missing tier shows in the configure output.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lapy3 tests")
  return()
endif()

# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case runs the
# kernel on the device.
calaman_add_gtest_executable(
  NAME
  lapy3_tests
  SOURCES
  lapy3_tests.cpp
  LINK
  calaman::lapy3
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
  lapy3_tests
  SUITES
  Lapy3OracleTests
  REQUIRES_GPU
)
