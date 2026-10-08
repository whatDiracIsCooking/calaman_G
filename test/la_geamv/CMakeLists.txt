# Oracle suite for calaman.la_geamv -- y := alpha * |op(A)| * |x| + beta * |y|
# with the symbolic-zero nudge. No LAPACKE wrapper exists and the image's
# liblapack does not export ?la_geamv_ (an XBLAS-only routine there), so the
# oracle is a host transcription of the Fortran loop in the test source (as
# test/larscl2/ does): no reference LAPACK, no guard.
#
# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case runs
# the kernel on the device.
calaman_add_gtest_executable(
  NAME
  la_geamv_tests
  SOURCES
  la_geamv_tests.cpp
  LINK
  calaman::la_geamv
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  la_geamv_tests
  SUITES
  LaGeamvOracleTests
  REQUIRES_GPU
)
