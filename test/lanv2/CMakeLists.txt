# Oracle test for calaman.lanv2 -- the Schur standardisation of a real 2-by-2.
# REQUIRES_GPU: every case stages the block on the device and copies it back, so
# `ctest -LE gpu` excludes it. It compares against the reference LAPACK ?lanv2,
# which arrives as calaman::lapack_reference; guard on that target and return
# early when it is absent -- a missing oracle is a missing TIER named in the
# configure output, not a silent pass (docs/architecture.md §3). ?lanv2 has no
# LAPACKE C binding (it is a LAPACK auxiliary routine), so the suite calls the
# Fortran slanv2_/dlanv2_ symbols directly. wwr.extension.memory_buffer plus the
# two test/shared/ modules stage the device block; lanv2 itself takes a bare
# stream and allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lanv2 tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lanv2_tests
  SOURCES
  lanv2_tests.cpp
  LINK
  calaman::lanv2
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
  lanv2_tests
  SUITES
  Lanv2OracleTests
  REQUIRES_GPU
)
