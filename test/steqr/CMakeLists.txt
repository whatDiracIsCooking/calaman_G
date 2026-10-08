# Oracle suite for calaman.steqr -- eigenpairs of a symmetric tridiagonal. The
# oracle is LAPACKE_?steqr, which arrives with calaman::lapack_reference; return
# early when absent so a missing oracle is a missing TIER named in the
# configure output, not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping steqr tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  steqr_tests
  SOURCES
  steqr_tests.cpp
  LINK
  calaman::steqr
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

# Every case stages (d, e, Z) on the device and runs the kernel, so it is
# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  steqr_tests
  SUITES
  SteqrOracleTests
  REQUIRES_GPU
)
