# Oracle suite for calaman.lacrm -- complex times real matrix multiply, c/z.
# The oracle is LAPACKE_clacrm / LAPACKE_zlacrm from calaman::lapack_reference;
# return early when absent so a missing oracle is a missing TIER named in the
# configure output, not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lacrm tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lacrm_tests
  SOURCES
  lacrm_tests.cpp
  LINK
  calaman::lacrm
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::blas
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

# Every case stages its operands on the device and runs the gemms, so it is
# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  lacrm_tests
  SUITES
  LacrmOracleTests
  REQUIRES_GPU
)
