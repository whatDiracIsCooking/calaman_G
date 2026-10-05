# Oracle test for calaman.lahr2 (the Hessenberg-panel reduction). REQUIRES_GPU:
# every case stages A on the device and runs the gemv/gemm/trmv/trmm kernels, so
# `ctest -LE gpu` excludes it. It compares against netlib's ?lahr2 through its
# Fortran ABI (this LAPACK build ships no LAPACKE_?lahr2 C binding -- ?lahr2 is
# an auxiliary routine), which arrives in LAPACK::LAPACK via
# calaman::lapack_reference; guard on that target and return early when it is
# absent -- a missing oracle is a missing tier named in the configure output,
# not a silent pass (see docs/architecture.md section 3). The two test/shared/
# modules plus wwr.extension.memory_buffer stage the device buffers; lahr2
# allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping lahr2 tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lahr2_tests
  SOURCES
  lahr2_tests.cpp
  LINK
  calaman::lahr2
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::test::shared::tolerance
  calaman::lapack_reference
  wwr::blas
  wwr::runtime_api
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  lahr2_tests
  SUITES
  Lahr2OracleTests
  REQUIRES_GPU
)
