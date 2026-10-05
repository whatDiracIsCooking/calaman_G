# Oracle test for calaman.laqr5 -- one non-accumulated (KACC22 = 0) multishift
# QR sweep with bulge chasing. REQUIRES_GPU: every case stages H / Z on the
# device and runs the kernel, so `ctest -LE gpu` excludes it. It compares with
# the reference LAPACK ?laqr5, which arrives as calaman::lapack_reference; guard
# on that target and return early when it is absent -- a missing oracle is a
# missing TIER named in the configure output, not a silent pass
# (docs/architecture.md §3). ?laqr5 has no LAPACKE C binding (it is a LAPACK
# auxiliary routine), so the suite calls the Fortran slaqr5_/dlaqr5_ symbols
# directly, pinned to KACC22 = 0 to match the device path. wwr.extension.memory_
# buffer plus the two test/shared/ modules stage the device data; laqr5 itself
# takes a bare stream and allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping laqr5 tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  laqr5_tests
  SOURCES
  laqr5_tests.cpp
  LINK
  calaman::laqr5
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
  laqr5_tests
  SUITES
  Laqr5OracleTests
  REQUIRES_GPU
)
