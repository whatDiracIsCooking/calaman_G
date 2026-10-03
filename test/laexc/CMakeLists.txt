# Oracle test for calaman.laexc -- swap two adjacent diagonal blocks of a real
# Schur form. REQUIRES_GPU: every case stages T / Q on the device and copies
# them back, so `ctest -LE gpu` excludes it. It compares against the reference
# LAPACK ?laexc, which arrives as calaman::lapack_reference; guard on the target
# and return early when absent -- a missing oracle is a missing TIER named in
# the configure output, not a silent pass (docs/architecture.md §3). ?laexc has
# no LAPACKE C binding (it is a computational routine), so the suite calls the
# Fortran slaexc_/dlaexc_ symbols directly. wwr.extension.memory_buffer plus the
# test/shared/ modules stage the device matrices; laexc itself takes a bare
# stream and allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping laexc tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  laexc_tests
  SOURCES
  laexc_tests.cpp
  LINK
  calaman.laexc
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
  laexc_tests
  SUITES
  LaexcOracleTests
  REQUIRES_GPU
)
