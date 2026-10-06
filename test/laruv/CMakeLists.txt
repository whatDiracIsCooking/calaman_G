# Oracle suite for calaman.laruv (built under src/lapack/laruv/), and through it
# the laruv_block header. The oracle is the Fortran slaruv_ / dlaruv_ (no
# LAPACKE binding), so the suite NEEDS calaman::lapack_reference; return early
# when absent so the missing tier shows in the configure output.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping laruv tests")
  return()
endif()

# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case runs the
# kernel on the device.
calaman_add_gtest_executable(
  NAME
  laruv_tests
  SOURCES
  laruv_tests.cpp
  LINK
  calaman::laruv
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  laruv_tests
  SUITES
  LaruvOracleTests
  REQUIRES_GPU
)
