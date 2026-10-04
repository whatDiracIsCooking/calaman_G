# Oracle test for calaman.laqr4 -- the small-bulge multishift QR driver with
# aggressive early deflation. REQUIRES_GPU: every case stages H / Z on the
# device and runs the laqr2 / lahqr / laqr5 kernels the driver launches, so
# `ctest -LE gpu` excludes it. It compares with the reference LAPACK ?laqr4,
# which arrives as calaman::lapack_reference; guard on that target and return
# early when it is absent -- a missing oracle is a missing TIER named in the
# configure output, not a silent pass (docs/architecture.md section 3). ?laqr4
# has no LAPACKE C binding (it is a LAPACK auxiliary routine), so the suite
# calls the Fortran slaqr4_/dlaqr4_ symbols directly. The two test/shared/
# modules and wwr.extension.memory_buffer stage the device data; laqr4 itself
# takes a bare stream and allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping laqr4 tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  laqr4_tests
  SOURCES
  laqr4_tests.cpp
  LINK
  calaman.laqr4
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
  laqr4_tests
  SUITES
  Laqr4OracleTests
  REQUIRES_GPU
)
