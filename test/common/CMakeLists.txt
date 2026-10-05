# Host-only suite for calaman.common's :workspace partition -- the workspace
# byte arithmetic and the carve-once idiom (slices_for + carve_workspace). No
# device memory, no kernel, no oracle: it links only calaman.common, is NOT
# marked REQUIRES_GPU, and runs on a card-less runner.
calaman_add_gtest_executable(
  NAME
  workspace_tests
  SOURCES
  workspace_tests.cpp
  LINK
  calaman::common
)

calaman_add_gtest_suite_tests(
  TARGET
  workspace_tests
  SUITES
  CommonWorkspaceTests
)
