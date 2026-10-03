# Oracle test for calaman.laqr2 -- one aggressive-early-deflation (AED) pass.
# REQUIRES_GPU: every case stages H / Z on the device and runs the kernel, so
# `ctest -LE gpu` excludes it. It compares with the reference LAPACK ?laqr2,
# which arrives as calaman::lapack_reference; guard on that target and return
# early when it is absent -- a missing oracle is a missing TIER named in the
# configure output, not a silent pass (docs/architecture.md §3). ?laqr2 has no
# LAPACKE C binding (it is a LAPACK auxiliary routine), so the suite calls the
# Fortran slaqr2_/dlaqr2_ symbols directly, and MUST use ?laqr2 (not ?laqr3) to
# match the device's small-window (?lahqr) variant. wwr.extension.memory_buffer
# plus the two test/shared/ modules stage the device data; laqr2 itself takes a
# bare stream and allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping laqr2 tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  laqr2_tests
  SOURCES
  laqr2_tests.cpp
  LINK
  calaman.laqr2
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
  laqr2_tests
  SUITES
  Laqr2OracleTests
  REQUIRES_GPU
)
