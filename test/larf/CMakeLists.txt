# Oracle test for calaman.larf (applying an elementary reflector).
# REQUIRES_GPU: every case stages C and v on the device and runs the gemv/ger
# kernels, so `ctest -LE gpu` excludes it. It compares against LAPACKE_?larf,
# which arrives as calaman::lapack_reference; guard on that target and return
# early when it is absent -- a missing oracle is a missing TIER named in the
# configure output, not a silent pass (docs/architecture.md §3). With
# wwr.extension.memory_buffer + the two test/shared/ modules staging the device
# buffers -- larf itself takes a bare wwrblasHandle_t plus a caller-supplied
# workspace and allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping larf tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  larf_tests
  SOURCES
  larf_tests.cpp
  LINK
  calaman.larf
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
  larf_tests
  SUITES
  LarfOracleTests
  REQUIRES_GPU
)
