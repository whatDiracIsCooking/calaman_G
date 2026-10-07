# Oracle suite for calaman.lagtm -- tridiagonal matrix-matrix multiply, s/d/c/z.
# The oracle is the reference ?lagtm, called as the Fortran symbol (neither
# LAPACKE nor lapack.h declares it), which arrives with
# calaman::lapack_reference; return early when absent so a missing oracle is a
# missing TIER named in the configure output, not a silent pass
# (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lagtm tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lagtm_tests
  SOURCES
  lagtm_tests.cpp
  LINK
  calaman::lagtm
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

# Every case stages its operands on the device and runs the kernel, so it is
# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  lagtm_tests
  SUITES
  LagtmOracleTests
  REQUIRES_GPU
)
