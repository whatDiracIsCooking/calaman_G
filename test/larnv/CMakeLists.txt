# Oracle suite for calaman.larnv (built under src/lapack/larnv/). The oracle is
# LAPACKE_?larnv, plus the Fortran dlaruv_ to read the multiplier table, so the
# suite NEEDS calaman::lapack_reference; return early when absent so the
# missing tier shows in the configure output.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping larnv tests")
  return()
endif()

# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case runs the
# kernels on the device.
calaman_add_gtest_executable(
  NAME
  larnv_tests
  SOURCES
  larnv_tests.cpp
  LINK
  calaman::larnv
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  larnv_tests
  SUITES
  LarnvOracleTests
  REQUIRES_GPU
)
