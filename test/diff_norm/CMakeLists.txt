# Oracle test for calaman.diff_norm (built under src/diff_norm/). REQUIRES_GPU
# labels the suite `gpu`, so `ctest -LE gpu` (the card-less CI presets) excludes
# it -- every case allocates device memory and runs the BLAS routines.
#
# The suite compares against the reference CBLAS (cblas_?asum / cblas_?nrm2 /
# cblas_i?amax), which arrives with the same oracle target as LAPACKE, so it
# cannot be built without the oracle. Guard on calaman::lapack_reference and
# return early when it is absent: a missing oracle is a missing TIER named in
# the configure output, not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping diff_norm tests"
  )
  return()
endif()

# diff_norm returns a single host scalar (host pointer mode), so the check is
# scalar-vs-scalar. wwr.extension.memory_buffer and the two test/shared/ modules
# are only for the device buffers that stage the inputs -- diff_norm itself
# takes a bare wwrblasHandle_t and allocates nothing.
calaman_add_gtest_executable(
  NAME
  diff_norm_tests
  SOURCES
  diff_norm_tests.cpp
  LINK
  calaman::diff_norm
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::blas
  wwr::runtime_api
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  diff_norm_tests
  SUITES
  DiffNormOracleTests
  REQUIRES_GPU
)
