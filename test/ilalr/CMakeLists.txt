# Oracle suite for calaman.ilalr -- the last non-zero row (ila?lr). The oracle
# is the reference Fortran (ilaslr_ & co., hand-declared: LAPACKE has no
# binding), linked through calaman::lapack_reference. Guard on it and return
# early when absent: a missing oracle is a missing TIER named in the configure
# output, not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping ilalr tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  ilalr_tests
  SOURCES
  ilalr_tests.cpp
  LINK
  calaman::ilalr
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
  ilalr_tests
  SUITES
  IlalrOracleTests
  REQUIRES_GPU
)
