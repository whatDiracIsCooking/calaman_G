# Oracle suite for calaman.geqp3 -- the top-level QR-with-column-pivoting
# driver, the all-free path (Geqp3OracleTests) and the caller-fixed-prefix path
# (Geqp3FixedPrefixTests). REQUIRES_GPU: every case stages A/tau/vn1/vn2/work on
# the device and runs the reflector + downdate kernels, so `ctest -LE gpu`
# excludes it. The oracle is LAPACKE_?geqp3 -- with jpvt = 0 for the all-free
# suites, and with a non-trivial input jpvt for the fixed-prefix suite (LAPACKE
# honours the same input convention) -- which arrives as
# calaman::lapack_reference; guard on that target and return early when it is
# absent -- a missing oracle is a missing TIER named in the configure output,
# not a silent pass (docs/architecture.md §3). The two test/shared/ modules
# stage the device buffers and tolerance supplies the factorization bound. geqp3
# takes a bare wwrblasHandle_t (and a wwrsolverDnHandle_t for the fixed-prefix
# overload, hence wwr.solver here for the test's own handle creation) plus
# caller-supplied device scratch, and allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping geqp3 tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  geqp3_tests
  SOURCES
  geqp3_tests.cpp
  LINK
  calaman.geqp3
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.utils.shared_device
  calaman.test.shared.tolerance
  calaman::lapack_reference
  wwr.blas
  wwr.solver
  wwr.runtime_api
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  geqp3_tests
  SUITES
  Geqp3OracleTests
  Geqp3FixedPrefixTests
  REQUIRES_GPU
)
