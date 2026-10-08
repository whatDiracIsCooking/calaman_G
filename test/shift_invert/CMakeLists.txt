# Oracle suite for calaman.shift_invert -- DenseShiftInvert, the
# (A - sigma I)^{-1} linear_operator. REQUIRES_GPU: every case factors and
# solves on the device, so `ctest -LE gpu` excludes it. The oracle is
# LAPACKE_?sysv on the shifted matrix (and LAPACKE_dsyev to place sigma), which
# arrives through calaman::lapack_reference; guard on that target and return
# early when it is absent -- a missing oracle is a missing TIER, not a silent
# pass (docs/architecture.md).
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping shift_invert tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  shift_invert_tests
  SOURCES
  shift_invert_tests.cpp
  LINK
  calaman::shift_invert
  calaman::common
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::test::shared::tolerance
  calaman::lapack_reference
  wwr::blas
  wwr::solver
  wwr::runtime_api
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  shift_invert_tests
  SUITES
  ShiftInvertOracleTests
  REQUIRES_GPU
)
