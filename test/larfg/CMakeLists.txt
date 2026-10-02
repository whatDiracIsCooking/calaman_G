# Oracle suite for calaman.larfg -- the elementary Householder reflector.
# It stages [alpha; x] on the device and runs the BLAS routines, so it is
# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`). It compares against
# LAPACKE_?larfg, which arrives as calaman::lapack_reference; guard on that
# target and return early when it is absent -- a missing oracle is a missing
# TIER named in the configure output, not a silent pass (docs/architecture.md
# §3). The two test/shared/ modules back the device buffers that stage the
# inputs, and tolerance supplies eps<T>().
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping larfg tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  larfg_tests
  SOURCES
  larfg_tests.cpp
  LINK
  calaman.larfg
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.shared.tolerance
  calaman.test.utils.shared_device
  calaman::lapack_reference
  wwr.blas
  wwr.runtime_api
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  larfg_tests
  SUITES
  LarfgOracleTests
  REQUIRES_GPU
)
