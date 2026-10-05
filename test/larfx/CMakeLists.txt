# Oracle test for calaman.larfx (applying an elementary reflector, fused for
# small orders). REQUIRES_GPU: every case stages C and v on the device and runs
# the fused kernel or the gemv/ger fallback, so `ctest -LE gpu` excludes it. It
# compares against netlib's ?larfx (Fortran ABI), which arrives through
# calaman::lapack_reference; guard on that target and return early when it is
# absent -- a missing oracle is a missing TIER named in the configure output,
# not a silent pass (docs/architecture.md §3). With wwr.extension.memory_buffer
# + the test/shared/ modules staging the device buffers -- larfx itself takes a
# bare wwrblasHandle_t plus a caller-supplied workspace and allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping larfx tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  larfx_tests
  SOURCES
  larfx_tests.cpp
  LINK
  calaman::larfx
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
  larfx_tests
  SUITES
  LarfxOracleTests
  REQUIRES_GPU
)
