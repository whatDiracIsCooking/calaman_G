# Oracle suite for calaman.gehrd -- the blocked reduction to upper Hessenberg
# form. It stages A/tau/work on the device and runs the panel reflector kernels,
# so it is REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`). It checks
# against reference ?gehrd through the Fortran symbol directly (as the gehd2 /
# lahr2 suites do), so it still needs the reference LAPACK linked, which arrives
# as calaman::lapack_reference; guard on it and return early when absent,
# a missing oracle is a missing tier named in the configure output, not a silent
# pass (docs/architecture.md §3). The three test/shared/ modules back the device
# buffers and supply eps<T>() for the tolerance.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping gehrd tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  gehrd_tests
  SOURCES
  gehrd_tests.cpp
  LINK
  calaman.gehrd
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
  gehrd_tests
  SUITES
  GehrdOracleTests
  REQUIRES_GPU
)
