# Oracle suite for calaman.lacp2 (built under src/lapack/lacp2/) -- copy a real
# matrix into a complex one. The oracle is LAPACKE_clacp2 / zlacp2, so the
# suite NEEDS calaman::lapack_reference; return early when absent so the
# missing tier shows in the configure output.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lacp2 tests")
  return()
endif()

# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case runs the
# copy kernel on the device.
calaman_add_gtest_executable(
  NAME
  lacp2_tests
  SOURCES
  lacp2_tests.cpp
  LINK
  calaman::lacp2
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
  lacp2_tests
  SUITES
  Lacp2OracleTests
  REQUIRES_GPU
)
