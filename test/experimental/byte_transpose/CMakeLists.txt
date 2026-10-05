# Suite for calaman.experimental.byte_transpose. Host-only (no device memory, no
# kernel, no oracle), so it is NOT marked REQUIRES_GPU and runs on a card-less
# CI runner. Reached only under CALAMAN_BUILD_EXPERIMENTAL (test/CMakeLists.txt
# gates the add_subdirectory), so it never configures in a default build.
calaman_add_gtest_executable(
  NAME
  byte_transpose_tests
  SOURCES
  byte_transpose_tests.cpp
  LINK
  calaman::experimental::byte_transpose
)

calaman_add_gtest_suite_tests(
  TARGET
  byte_transpose_tests
  SUITES
  ByteTransposeTest
)
