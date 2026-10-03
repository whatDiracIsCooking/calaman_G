# Oracle test for calaman.trevc3 -- the left/right eigenvectors of a real
# quasi-triangular (Schur) matrix, HOWMNY = 'A'. REQUIRES_GPU: every case stages
# T on the device and runs the kernel, so `ctest -LE gpu` excludes it. It
# compares with reference LAPACK ?trevc3, which arrives as
# calaman::lapack_reference; guard on that target and return early when it is
# absent -- a missing oracle is a missing TIER named in the configure output,
# not a silent pass (docs/architecture.md §3). ?trevc3 has no LAPACKE C binding
# (a LAPACK auxiliary routine), so the suite calls the Fortran strevc3_ /
# dtrevc3_ symbols directly. The primary assertion is the scale/sign-invariant
# eigen-residual; a sign-resolved column comparison against the reference is the
# tighter cross-check. wwr.extension.memory_buffer plus the two test/shared/
# modules stage the device data; trevc3 itself takes a bare stream and allocates
# nothing.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping trevc3 tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  trevc3_tests
  SOURCES
  trevc3_tests.cpp
  LINK
  calaman.trevc3
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
  trevc3_tests
  SUITES
  Trevc3OracleTests
  REQUIRES_GPU
)
