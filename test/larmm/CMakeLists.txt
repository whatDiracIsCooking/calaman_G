# Oracle suite for calaman.larmm (built under src/lapack/larmm/), and through it
# the header-only larmm_scalar in larmm.cuh. The oracle is the Fortran slarmm_ /
# dlarmm_ (no LAPACKE binding), so the suite NEEDS calaman::lapack_reference;
# return early when absent so the missing tier shows in the configure output.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping larmm tests")
  return()
endif()

# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case runs the
# kernel on the device.
calaman_add_gtest_executable(
  NAME
  larmm_tests
  SOURCES
  larmm_tests.cpp
  LINK
  calaman::larmm
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  larmm_tests
  SUITES
  LarmmOracleTests
  REQUIRES_GPU
)
