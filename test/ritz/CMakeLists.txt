# Suite for calaman.ritz: ritz_select and classify_ritz (pure host), the
# argument checks of ritz_rotate, and ritz_rotate's product on the device
# against a host product. No reference LAPACK: the inputs are Householder
# reflectors, orthonormal by construction.

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
