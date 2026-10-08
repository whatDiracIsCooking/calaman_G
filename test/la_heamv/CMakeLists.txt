# Oracle suite for calaman.la_heamv -- y := alpha * |A| * |x| + beta * |y| for
# a Hermitian A held in one triangle, with the symbolic-zero nudge. The oracle
# is a host transcription of the loop, as for test/la_syamv/ (the image's
# liblapack does not export ?la_heamv_). It also links calaman.la_syamv to
# assert the shared kernel is shared: both agree bit for bit on one triangle.
#
# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case runs
# the kernel on the device.
calaman_add_gtest_executable(
  NAME
  la_heamv_tests
  SOURCES
  la_heamv_tests.cpp
  LINK
  calaman::la_heamv
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
  la_heamv_tests
  SUITES
  LaHeamvOracleTests
  REQUIRES_GPU
)
