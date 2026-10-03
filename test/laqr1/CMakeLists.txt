# Oracle test for calaman.laqr1 -- the scaled first column of the shift
# polynomial (H-s1*I)(H-s2*I). REQUIRES_GPU: every case stages H on the device
# and copies V back, so `ctest -LE gpu` excludes it. It compares against the
# reference LAPACK ?laqr1, which arrives as calaman::lapack_reference; guard on
# that target and return early when it is absent -- a missing oracle is a
# missing TIER named in the configure output, not a silent pass
# (docs/architecture.md §3). ?laqr1 has no LAPACKE C binding (it is a LAPACK
# auxiliary routine), so the suite calls the Fortran slaqr1_/dlaqr1_ symbols
# directly. wwr.extension.memory_buffer plus the two test/shared/ modules stage
# the device data; laqr1 itself takes a bare stream and allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping laqr1 tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  laqr1_tests
  SOURCES
  laqr1_tests.cpp
  LINK
  calaman.laqr1
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.shared.tolerance
  calaman.test.utils.shared_device
  calaman::lapack_reference
  wwr.runtime_api
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  laqr1_tests
  SUITES
  Laqr1OracleTests
  REQUIRES_GPU
)
