# Oracle suite for calaman.ormhr -- multiply a general matrix by the orthogonal
# Q from ?gehrd (Q*C, Q^T*C, C*Q, C*Q^T). REQUIRES_GPU: every case stages A/tau/
# C/work on the device and runs the ormqr kernel, so `ctest -LE gpu` skips it.
# The oracle is LAPACKE_?ormhr, with the reflectors from LAPACKE_?gehrd -- both
# arrive through calaman::lapack_reference; guard on that target and return
# early when absent, a missing oracle is a missing TIER named in the configure
# output, not a silent pass (docs/architecture.md §3). The three test/shared/
# modules stage the device buffers and supply the factorization tolerance. ormhr
# takes a wwrsolverDnHandle_t (its work is entirely ormqr), hence wwr.solver
# for the test's own handle creation; calaman.common supplies Side/Trans.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping ormhr tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  ormhr_tests
  SOURCES
  ormhr_tests.cpp
  LINK
  calaman.ormhr
  calaman.common
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
  ormhr_tests
  SUITES
  OrmhrOracleTests
  REQUIRES_GPU
)
