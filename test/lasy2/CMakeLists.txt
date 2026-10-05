# Oracle suite for calaman.lasy2 -- the order-(1..2) Sylvester solve. It stages
# the matrices on the device and runs the kernel, so it is REQUIRES_GPU (labeled
# `gpu`, excluded by `ctest -LE gpu`). It compares against reference ?lasy2,
# which has no LAPACKE binding, so the suite calls the Fortran symbol directly
# -- but it still needs the reference LAPACK linked, which arrives as
# calaman::lapack_reference; guard on that target and return early when absent,
# a missing oracle is a missing tier named in the configure output, not a silent
# pass (docs/architecture.md §3). The three test/shared/ modules back the device
# buffers and supply eps<T>() for the tolerance.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lasy2 tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lasy2_tests
  SOURCES
  lasy2_tests.cpp
  LINK
  calaman::lasy2
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
  lasy2_tests
  SUITES
  Lasy2OracleTests
  REQUIRES_GPU
)
