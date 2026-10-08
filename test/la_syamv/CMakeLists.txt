# Oracle suite for calaman.la_syamv -- y := alpha * |A| * |x| + beta * |y| for
# a symmetric A held in one triangle, with the symbolic-zero nudge. No LAPACKE
# wrapper exists and the image's liblapack does not export ?la_syamv_ (an
# XBLAS-only routine there), so the oracle is a host transcription of the loop
# in the test source, as for test/la_geamv/: no reference LAPACK, no guard.
#
# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case runs
# the kernel on the device.
calaman_add_gtest_executable(
  NAME
  la_syamv_tests
  SOURCES
  la_syamv_tests.cpp
  LINK
  calaman::la_syamv
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  la_syamv_tests
  SUITES
  LaSyamvOracleTests
  REQUIRES_GPU
)
