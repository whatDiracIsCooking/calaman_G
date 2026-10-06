# Oracle suite for calaman.syev -- the real symmetric eigensolver driver. The
# oracle is LAPACKE_?syev, which arrives with calaman::lapack_reference; return
# early when absent so a missing oracle is a missing TIER named in the
# configure output, not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping syev tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  syev_tests
  SOURCES
  syev_tests.cpp
  LINK
  calaman::syev
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::solver
  wwr::extension::memory_buffer
)

# Every case runs the driver on the device, so it is REQUIRES_GPU (labeled
# `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  syev_tests
  SUITES
  SyevOracleTests
  REQUIRES_GPU
)
