# Oracle suite for calaman.lat2 (built under src/lapack/lat2/) -- triangular
# matrix, double to single. The oracle is the Fortran dlat2s_ / zlat2c_ (no
# LAPACKE binding), so the suite NEEDS calaman::lapack_reference; return early
# when absent so the missing tier shows in the configure output.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lat2 tests")
  return()
endif()

# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case runs the
# conversion kernel on the device.
calaman_add_gtest_executable(
  NAME
  lat2_tests
  SOURCES
  lat2_tests.cpp
  LINK
  calaman::lat2
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  lat2_tests
  SUITES
  Lat2OracleTests
  REQUIRES_GPU
)
