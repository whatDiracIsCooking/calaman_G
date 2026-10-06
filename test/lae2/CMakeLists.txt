# Oracle suite for calaman.lae2 (built under src/lapack/lae2/), and through it
# the shared lae2_scalar in sym2x2.cuh. The oracle is the Fortran slae2_ /
# dlae2_ (no LAPACKE binding), so the suite NEEDS calaman::lapack_reference;
# return early when absent so the missing tier shows in the configure output.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lae2 tests")
  return()
endif()

# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case runs the
# kernel on the device.
calaman_add_gtest_executable(
  NAME
  lae2_tests
  SOURCES
  lae2_tests.cpp
  LINK
  calaman::lae2
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
  lae2_tests
  SUITES
  Lae2OracleTests
  REQUIRES_GPU
)
