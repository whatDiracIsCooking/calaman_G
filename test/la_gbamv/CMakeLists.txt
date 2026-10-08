# Oracle suite for calaman.la_gbamv -- y := alpha * |op(A)| * |x| + beta * |y|
# for a band A, with the symbolic-zero nudge. No LAPACKE wrapper exists and the
# image's liblapack does not export ?la_gbamv_ (an XBLAS-only routine there),
# so the oracle is a host transcription of the loop in the test source, as for
# test/la_geamv/: no reference LAPACK, no guard.
#
# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case runs
# the kernel on the device.
calaman_add_gtest_executable(
  NAME
  la_gbamv_tests
  SOURCES
  la_gbamv_tests.cpp
  LINK
  calaman::la_gbamv
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  la_gbamv_tests
  SUITES
  LaGbamvOracleTests
  REQUIRES_GPU
)
