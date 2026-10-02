# Oracle suite for calaman.columnwise_ell2 -- per-column L2 norm. There is no
# LAPACKE binding for a batched per-column L2 norm; the oracle is the reference
# CBLAS cblas_?nrm2 applied to each column (cblas.h arrives with the same
# calaman::lapack_reference target as LAPACKE). Guard on it and return early
# when absent: a missing oracle is a missing TIER named in the configure output,
# not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping columnwise_ell2 tests"
  )
  return()
endif()

# The test/shared/ modules back the device buffers that stage A and the result;
# columnwise_ell2 itself takes a bare wwrStream_t and allocates nothing.
calaman_add_gtest_executable(
  NAME
  columnwise_ell2_tests
  SOURCES
  columnwise_ell2_tests.cpp
  LINK
  calaman.columnwise_ell2
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.utils.shared_device
  calaman::lapack_reference
  wwr.runtime_api
  wwr.extension.memory_buffer
)

# Two registrations for one target (the drift guard unions them): the
# empty-input contract is a pure host call -- columnwise_ell2 returns before
# touching the stream or a device pointer, so it takes a null stream and host
# dummy pointers
# and runs on a card-less runner. The oracle suite stages A on the device and
# runs the reduction kernel, so it is REQUIRES_GPU (labeled `gpu`, excluded by
# ctest -LE gpu).
calaman_add_gtest_suite_tests(
  TARGET
  columnwise_ell2_tests
  SUITES
  ColumnwiseEll2ArgCheckTests
)

calaman_add_gtest_suite_tests(
  TARGET
  columnwise_ell2_tests
  SUITES
  ColumnwiseEll2OracleTests
  REQUIRES_GPU
)
