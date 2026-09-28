# Self-test for the calaman.test.elementwise_compare utility (built under
# test/utils/elementwise_compare/). REQUIRES_GPU labels the suite `gpu`, so
# `ctest -LE gpu` (the card-less CI presets) excludes it -- every case launches
# the two kernels.
calaman_add_gtest_executable(
  NAME
  elementwise_compare_tests
  SOURCES
  elementwise_compare_tests.cpp
  LINK
  calaman.test.elementwise_compare
  wwr.runtime_api
  wwr.extension.error_handling
  wwr.extension.runtime
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  elementwise_compare_tests
  SUITES
  ElementwiseCompareTests
  REQUIRES_GPU
)
