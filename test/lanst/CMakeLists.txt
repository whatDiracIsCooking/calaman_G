# Oracle suite for calaman.lanst -- the ?lanst tridiagonal norm, and the
# header-only lanst_max_abs. The oracle is the reference ?lanst, called through
# lapack.h (LAPACKE wraps no ?lanst), which arrives with
# calaman::lapack_reference; return early when absent so a missing oracle is a
# missing TIER named in the configure output, not a silent pass
# (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lanst tests")
  return()
endif()

# calaman::lanst::header is the downstream spelling of lapack/lanst/lanst.h,
# linked here exactly as a ?sterf/?steqr device library would.
calaman_add_gtest_executable(
  NAME
  lanst_tests
  SOURCES
  lanst_tests.cpp
  LINK
  calaman::lanst
  calaman::lanst::header
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::extension::memory_buffer
)

# LanstOracleTests stages (d, e) on the device -- even n == 0, a device memset
# -- so it is REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  lanst_tests
  SUITES
  LanstOracleTests
  REQUIRES_GPU
)

# LanstHelperTests calls lanst_max_abs on the host only: no device needed.
calaman_add_gtest_suite_tests(
  TARGET
  lanst_tests
  SUITES
  LanstHelperTests
)
