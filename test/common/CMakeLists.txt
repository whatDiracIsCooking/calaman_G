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

# GPU suite for common/warp_reduce.cuh. The kernels .cu is a device pass on
# both backends, so it is its own device library; the gtest .cpp stays host
# C++ and reaches it only through warp_reduce_bridge.h's plain-type launchers.
calaman_add_gpu_device_library(
  NAME
  warp_reduce_tests.device
  SOURCES
  warp_reduce_kernels.cu
  LINK_PRIVATE
  wwr::device
  calaman::common
)

calaman_add_gtest_executable(
  NAME
  warp_reduce_tests
  SOURCES
  warp_reduce_tests.cpp
  LINK
  warp_reduce_tests.device
)

calaman_add_gtest_suite_tests(
  TARGET
  warp_reduce_tests
  SUITES
  CommonWarpReduceTests
  REQUIRES_GPU
)
