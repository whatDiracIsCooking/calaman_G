# Oracle test for calaman.trexc -- reorder a real Schur form by moving a
# selected diagonal block. REQUIRES_GPU: every case stages T / Q on the device
# and copies
# them back, so `ctest -LE gpu` excludes it. It compares against the reference
# LAPACK ?trexc, which arrives as calaman::lapack_reference; guard on the target
# and return early when absent -- a missing oracle is a missing TIER named in
# the configure output, not a silent pass (docs/architecture.md §3). ?trexc has
# a LAPACKE C binding, so the suite calls LAPACKE_strexc / LAPACKE_dtrexc direct
# (column-major), unlike the laexc suite's Fortran-symbol dance.
# wwr.extension.memory_buffer plus the test/shared/ modules stage the device
# matrices; trexc itself takes a bare stream and allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping trexc tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  trexc_tests
  SOURCES
  trexc_tests.cpp
  LINK
  calaman::trexc
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  trexc_tests
  SUITES
  TrexcOracleTests
  REQUIRES_GPU
)
