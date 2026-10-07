# Suite for calaman.ritz: ritz_select and classify_ritz (pure host), the
# argument checks of ritz_rotate, and ritz_rotate's product against the
# reference LAPACK.

# Host only: selection and classification are host math, and a bad
# ritz_rotate argument is rejected before any handle use, so these run on a
# card-less runner (not REQUIRES_GPU).
calaman_add_gtest_executable(
  NAME
  ritz_tests
  SOURCES
  ritz_tests.cpp
  LINK
  calaman::ritz
  wwr::blas
)

calaman_add_gtest_suite_tests(
  TARGET
  ritz_tests
  SUITES
  RitzSelectTests
  RitzClassifyTests
  RitzRotateArgTests
)

# The rotation's oracle is LAPACKE_?syevd plus ?geqrf/?orgqr, via
# calaman::lapack_reference. Missing is a WARNING, not an error, so only this
# binary is skipped where it is absent.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping ritz_rotate_tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  ritz_rotate_tests
  SOURCES
  ritz_rotate_tests.cpp
  LINK
  calaman::ritz
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::blas
  wwr::runtime_api
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  ritz_rotate_tests
  SUITES
  RitzRotateReferenceTests
  REQUIRES_GPU
)
