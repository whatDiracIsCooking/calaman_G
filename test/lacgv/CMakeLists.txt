# Oracle suite for calaman.lacgv -- conjugate a complex vector. It stages the
# strided buffer on the device and runs the kernel, so it is REQUIRES_GPU
# (labeled `gpu`, excluded by `ctest -LE gpu`). It compares against
# LAPACKE_?lacgv, which arrives as calaman::lapack_reference; guard on that
# target and return early when absent -- a missing oracle is a missing tier
# named in the configure output, not a silent pass (see docs/architecture.md).
# The two test/shared/ modules back the device buffers that stage the inputs;
# there is no tolerance module, because conjugation is bit-exact.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping lacgv tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lacgv_tests
  SOURCES
  lacgv_tests.cpp
  LINK
  calaman.lacgv
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.utils.shared_device
  calaman::lapack_reference
  wwr.runtime_api
  wwr.complex
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  lacgv_tests
  SUITES
  LacgvOracleTests
  REQUIRES_GPU
)
