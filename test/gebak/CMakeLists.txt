# Oracle suite for calaman.gebak -- backward transformation of eigenvectors.
# gebak has no balancing-convention divergence from a reference LAPACK (it only
# undoes a scaling and a permutation), so LAPACKE_?gebak is a true oracle and
# this is an oracle diff, not a spec test like gebal. The oracle arrives as
# calaman::lapack_reference; guard on that target and return early when it is
# absent -- a missing oracle is a missing TIER named in the configure output,
# not a silent pass (docs/architecture.md §3).
#
# REQUIRES_GPU: every case stages V/scale on the device and runs the
# scale/permute kernels, so `ctest -LE gpu` excludes it. The two test/shared/
# modules stage the device buffers; wwr.complex and wwr.wrappers.common supply
# the complex element types and the ComplexToRealType the scale array uses.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping gebak tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  gebak_tests
  SOURCES
  gebak_tests.cpp
  LINK
  calaman::gebak
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::wrappers::common
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  gebak_tests
  SUITES
  GebakOracleTests
  REQUIRES_GPU
)
