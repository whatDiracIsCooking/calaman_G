# Oracle suite for calaman.lag2 (built under src/lapack/lag2/) -- general
# matrix, double <-> single. The oracle is LAPACKE_dlag2s / slag2d / zlag2c /
# clag2z, so the suite NEEDS calaman::lapack_reference; return early when
# absent so the missing tier shows in the configure output.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lag2 tests")
  return()
endif()

# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case runs the
# conversion kernel on the device.
calaman_add_gtest_executable(
  NAME
  lag2_tests
  SOURCES
  lag2_tests.cpp
  LINK
  calaman::lag2
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
  lag2_tests
  SUITES
  Lag2OracleTests
  REQUIRES_GPU
)
