# Oracle suite for calaman.lassq (built under src/lapack/lassq/) -- the ?lassq
# scaled sum of squares of a vector. The oracle is the reference ?lassq, called
# through lapack.h's LAPACK_?lassq (LAPACKE wraps no ?lassq), which arrives with
# calaman::lapack_reference; return early when absent so a missing oracle is a
# missing TIER named in the configure output, not a silent pass
# (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lassq tests")
  return()
endif()

# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case stages
# the vector and the (scale, sumsq) pair on the device and runs the kernel. The
# staging buffers come from wwr.extension.memory_buffer and the test/shared/
# modules; lassq itself takes a bare wwrStream_t and allocates nothing.
calaman_add_gtest_executable(
  NAME
  lassq_tests
  SOURCES
  lassq_tests.cpp
  LINK
  calaman::lassq
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  lassq_tests
  SUITES
  LassqOracleTests
  REQUIRES_GPU
)
