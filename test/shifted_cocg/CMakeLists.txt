# Suite for calaman.shifted_cocg: the argument checks and workspace sizing, and
# the solve against LAPACKE_zgesv on each z_e I - A (LAPACKE_dsyevd for the
# condition numbers), through a symm-backed operator: wwr::wrappers::blas
# supplies symm. The oracle comes from calaman::lapack_reference; missing is a
# WARNING, not an error, so the whole suite is skipped where it is absent.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping shifted_cocg_tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  shifted_cocg_tests
  SOURCES
  shifted_cocg_tests.cpp
  LINK
  calaman::shifted_cocg
  calaman::test::shared::abort_policy
  calaman::test::shared::expect_converged
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::blas
  wwr::runtime_api
  wwr::wrappers::blas
  wwr::extension::memory_buffer
)

# Pure host calls: every bad argument is rejected before the stream or the
# operator is touched, and sizing allocates nothing.
calaman_add_gtest_suite_tests(
  TARGET
  shifted_cocg_tests
  SUITES
  ShiftedCocgArgCheckTests
)

calaman_add_gtest_suite_tests(
  TARGET
  shifted_cocg_tests
  SUITES
  ShiftedCocgReferenceTests
  REQUIRES_GPU
)
