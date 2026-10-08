# Oracle suite for calaman.larscl2 -- reciprocal diagonal row-scaling,
# X <-- D^-1 X. No LAPACKE wrapper exists and the image's liblapack does not
# export ?larscl2_, so the oracle is a host transcription of the Fortran loop in
# the test source (as test/lascl2/ does): no reference LAPACK, no guard.
#
# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case runs
# the kernel on the device.
calaman_add_gtest_executable(
  NAME
  larscl2_tests
  SOURCES
  larscl2_tests.cpp
  LINK
  calaman::larscl2
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  larscl2_tests
  SUITES
  Larscl2OracleTests
  REQUIRES_GPU
)
