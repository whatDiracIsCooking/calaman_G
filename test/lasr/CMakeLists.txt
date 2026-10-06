# Oracle suite for calaman.lasr -- apply a sequence of plane rotations. The
# oracle is the reference ?lasr, called as the Fortran symbol (neither LAPACKE
# nor lapack.h declares it), which arrives with calaman::lapack_reference;
# return early when absent so a missing oracle is a missing TIER named in the
# configure output, not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lasr tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lasr_tests
  SOURCES
  lasr_tests.cpp
  LINK
  calaman::lasr
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::extension::memory_buffer
)

# Every case stages A on the device and runs the kernel, so it is REQUIRES_GPU
# (labeled `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  lasr_tests
  SUITES
  LasrOracleTests
  REQUIRES_GPU
)
