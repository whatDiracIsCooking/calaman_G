# Oracle test for calaman.larf1f (applying a reflector with an implicit unit
# v(1)). REQUIRES_GPU: every case stages C and v on the device and runs the
# gemv/ger/axpy/scal kernels, so `ctest -LE gpu` excludes it. It compares
# against netlib ?larf (dlarf_/slarf_, with v(1) forced to 1 -- see
# larf1f_tests.cpp), which arrives as calaman::lapack_reference; guard on that
# target and return early when it is absent -- a missing oracle is a missing
# TIER named in the configure output, not a silent pass (docs/architecture.md
# §3). With wwr.extension.memory_buffer + the two test/shared/ modules staging
# the device buffers -- larf1f itself takes a bare wwrblasHandle_t plus a
# caller-supplied workspace and allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping larf1f tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  larf1f_tests
  SOURCES
  larf1f_tests.cpp
  LINK
  calaman.larf1f
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.utils.shared_device
  calaman.test.shared.tolerance
  calaman::lapack_reference
  wwr.blas
  wwr.runtime_api
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  larf1f_tests
  SUITES
  Larf1fOracleTests
  REQUIRES_GPU
)
