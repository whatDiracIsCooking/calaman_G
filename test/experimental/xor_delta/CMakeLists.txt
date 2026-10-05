# Suite for calaman.experimental.xor_delta. Host-only (no device memory, no
# kernel, no oracle), so it is NOT marked REQUIRES_GPU and runs on a card-less
# CI runner. Reached only under CALAMAN_BUILD_EXPERIMENTAL (test/CMakeLists.txt
# gates the add_subdirectory), so it never configures in a default build.
calaman_add_gtest_executable(
  NAME
  xor_delta_tests
  SOURCES
  xor_delta_tests.cpp
  LINK
  calaman::experimental::xor_delta
)

calaman_add_gtest_suite_tests(
  TARGET
  xor_delta_tests
  SUITES
  XorDeltaTests
)
