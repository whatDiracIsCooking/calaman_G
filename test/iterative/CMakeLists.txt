# Suite for calaman.iterative -- the stop_reason concept, IterationInfo and
# converged(), plus the EXPECT_CONVERGED helper. Host-only (no device, no
# oracle), so it is NOT marked REQUIRES_GPU and runs on a card-less CI runner.
calaman_add_gtest_executable(
  NAME
  iterative_tests
  SOURCES
  iterative_tests.cpp
  LINK
  calaman::iterative
  calaman::test::shared::expect_converged
)

calaman_add_gtest_suite_tests(
  TARGET
  iterative_tests
  SUITES
  IterativeTests
)
