# Oracle suite for calaman.lasrt (built under src/lasrt/) -- sort a 1-D array
# into increasing or decreasing order. The oracle is the reference
# LAPACKE_?lasrt run over the identical input; a sort reorders values without
# computing any, so the device and reference agree bit for bit and the suite
# compares with `==`.
#
# It cannot be built without the oracle, so guard on calaman::lapack_reference
# and return early when it is absent: a missing oracle is a missing TIER named
# in the configure output, not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lasrt tests")
  return()
endif()

# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every sorting case
# stages the array on the device and runs the sort kernel.
# wwr.extension.memory_buffer and the two test/shared/ modules back the device
# buffers that stage inputs; lasrt itself takes a bare wwrStream_t and allocates
# nothing.
calaman_add_gtest_executable(
  NAME
  lasrt_tests
  SOURCES
  lasrt_tests.cpp
  LINK
  calaman.lasrt
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.utils.shared_device
  calaman::lapack_reference
  wwr.runtime_api
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  lasrt_tests
  SUITES
  LasrtOracleTests
  REQUIRES_GPU
)
