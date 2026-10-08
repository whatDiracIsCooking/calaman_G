# Oracle suite for calaman.la_wwaddw -- the double-word accumulate
# (x, y) <-- (x, y) + w. No LAPACKE wrapper exists and the image's liblapack
# does not export ?la_wwaddw_, so the oracle is a host port of the Fortran loop
# in the test source, compared bit for bit: no reference LAPACK, no guard.
#
# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case runs
# the kernel on the device.
calaman_add_gtest_executable(
  NAME
  la_wwaddw_tests
  SOURCES
  la_wwaddw_tests.cpp
  LINK
  calaman::la_wwaddw
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  la_wwaddw_tests
  SUITES
  LaWwaddwOracleTests
  REQUIRES_GPU
)
