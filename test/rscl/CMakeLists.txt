# Oracle suite for calaman.rscl (built under src/lapack/rscl/) -- scale a vector
# by 1/a without over/underflow. The oracle is the reference ?rscl family, whose
# four Fortran symbols the suite declares itself (neither LAPACKE nor lapack.h
# wraps ?rscl); they arrive with calaman::lapack_reference, so return early when
# it is absent -- a missing oracle is a missing TIER named in the configure
# output, not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping rscl tests")
  return()
endif()

# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case stages
# the vector on the device and runs the scaling kernel. The staging buffers come
# from wwr.extension.memory_buffer and the test/shared/ modules; rscl itself
# takes a bare wwrStream_t and allocates nothing. No tolerance module: both
# paths apply the identical factor chain, so the comparison is exact.
calaman_add_gtest_executable(
  NAME
  rscl_tests
  SOURCES
  rscl_tests.cpp
  LINK
  calaman::rscl
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  rscl_tests
  SUITES
  RsclOracleTests
  REQUIRES_GPU
)
