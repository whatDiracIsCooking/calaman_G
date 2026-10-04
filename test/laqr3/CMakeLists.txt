# Oracle test for calaman.laqr3 -- one aggressive-early-deflation (AED) pass,
# the recursive (?laqr4 window Schur form) variant. REQUIRES_GPU: every case
# stages H / Z on the device and runs the kernels (the calaman.laqr2 delegate on
# the small-window path, or laqr3_setup / calaman.laqr4 / laqr3_finish on the
# large), so `ctest -LE gpu` excludes it. It compares with the reference LAPACK
# ?laqr3, which arrives as calaman::lapack_reference; guard on that target and
# return early when it is absent -- a missing oracle is a missing TIER named in
# the configure output, not a silent pass (docs/architecture.md section 3).
# ?laqr3 has no LAPACKE C binding (it is a LAPACK auxiliary routine), so the
# suite calls the Fortran slaqr3_/dlaqr3_ symbols directly. Small windows
# (JW <= NMIN = 75) match element-wise (?laqr3 == ?laqr2 there); large windows
# check backend-stable invariants (eigenvalue set + reconstruction), since the
# ?laqr4 Schur basis is not unique. The two test/shared/ modules and
# wwr.extension.memory_buffer stage the device data; laqr3 itself takes a bare
# stream and allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping laqr3 tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  laqr3_tests
  SOURCES
  laqr3_tests.cpp
  LINK
  calaman.laqr3
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
  laqr3_tests
  SUITES
  Laqr3OracleTests
  REQUIRES_GPU
)
