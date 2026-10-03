# Oracle suite for calaman.lascl (built under src/lascl/) -- multiply a matrix
# by cto/cfrom without over/underflow (?lascl, TYPE='G'). The oracle is the
# reference LAPACKE_?lascl run over the identical inputs; both paths apply the
# same guarded multiplier chain, so the device and reference agree to rounding
# and the suite compares with a relative-or-absolute tolerance (not `==`).
#
# It cannot be built without the oracle, so guard on calaman::lapack_reference
# and return early when it is absent: a missing oracle is a missing TIER named
# in the configure output, not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lascl tests")
  return()
endif()

# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case stages
# the matrix on the device and runs the scaling kernel. The device buffers that
# stage inputs come from wwr.extension.memory_buffer and the two test/shared/
# modules;
# lascl itself takes a bare wwrStream_t and allocates nothing.
calaman_add_gtest_executable(
  NAME
  lascl_tests
  SOURCES
  lascl_tests.cpp
  LINK
  calaman.lascl
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.shared.tolerance
  calaman.test.utils.shared_device
  calaman::lapack_reference
  wwr.runtime_api
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  lascl_tests
  SUITES
  LasclOracleTests
  REQUIRES_GPU
)
