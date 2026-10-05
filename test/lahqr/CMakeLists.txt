# Oracle test for calaman.lahqr -- double-shift Francis QR for the Schur form of
# an upper Hessenberg matrix. REQUIRES_GPU: every case stages H / Z on the
# device and copies them back, so `ctest -LE gpu` excludes it. It compares
# against the reference LAPACK ?lahqr, which arrives as
# calaman::lapack_reference; guard on the target, return early when absent -- a
# missing TIER named in the configure output, not a silent pass
# (docs/architecture.md §3). ?lahqr has no LAPACKE C binding (it is an auxiliary
# routine), so the suite calls the Fortran slahqr_/dlahqr_ symbols directly.
# wwr.extension.memory_buffer plus
# the test/shared/ modules stage the device matrices; lahqr itself takes a bare
# stream and allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lahqr tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lahqr_tests
  SOURCES
  lahqr_tests.cpp
  LINK
  calaman::lahqr
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
  lahqr_tests
  SUITES
  LahqrOracleTests
  REQUIRES_GPU
)
