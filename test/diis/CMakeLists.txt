# Oracle suite for calaman.diis -- Pulay DIIS extrapolation over a device ring.
# The oracle is a host DIIS in double whose bordered system LAPACKE_dgesv
# solves, so it arrives through calaman::lapack_reference; guard on that target
# and return early when it is absent -- a missing oracle is a missing TIER, not
# a silent pass.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping diis tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  diis_tests
  SOURCES
  diis_tests.cpp
  LINK
  calaman::diis
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::test::shared::tolerance
  calaman::lapack_reference
  wwr::blas
  wwr::runtime_api
  wwr::extension::memory_buffer
)

# Every case runs the gemvs and the solve kernel on the device, so the suite is
# REQUIRES_GPU (`gpu`, excluded by ctest -LE gpu).
calaman_add_gtest_suite_tests(
  TARGET
  diis_tests
  SUITES
  DiisOracleTests
  REQUIRES_GPU
)
