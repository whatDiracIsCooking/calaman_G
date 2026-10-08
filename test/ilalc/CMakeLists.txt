# Oracle suite for calaman.ilalc -- the last non-zero column (ila?lc). The
# oracle is the reference Fortran (ilaslc_ & co., hand-declared: LAPACKE has no
# binding), linked through calaman::lapack_reference. Guard on it and return
# early when absent: a missing oracle is a missing TIER named in the configure
# output, not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping ilalc tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  ilalc_tests
  SOURCES
  ilalc_tests.cpp
  LINK
  calaman::ilalc
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

# Every case runs the scan on the device, so the suite is REQUIRES_GPU.
calaman_add_gtest_suite_tests(
  TARGET
  ilalc_tests
  SUITES
  IlalcOracleTests
  REQUIRES_GPU
)
