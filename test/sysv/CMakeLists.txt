# Oracle suite for calaman.sysv -- the symmetric indefinite solve driver.
# REQUIRES_GPU: every case factors (wwr::sytrf) and solves on the device, so
# `ctest -LE gpu` excludes it. The oracle is LAPACKE_?sysv plus a backward-error
# residual, which arrives through calaman::lapack_reference; guard on that
# target and return early when it is absent -- a missing oracle is a missing
# TIER, not a silent pass (docs/architecture.md). sysv takes a wwrblasHandle_t +
# a wwrsolverDnHandle_t + caller device arrays and allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping sysv tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  sysv_tests
  SOURCES
  sysv_tests.cpp
  LINK
  calaman.sysv
  calaman.common
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.utils.shared_device
  calaman.test.shared.tolerance
  calaman::lapack_reference
  wwr.blas
  wwr.solver
  wwr.runtime_api
  wwr.complex
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  sysv_tests
  SUITES
  SysvOracleTests
  REQUIRES_GPU
)
