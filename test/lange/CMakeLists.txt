# Oracle suite for calaman.lange -- the ?lange matrix norm. The oracle is
# LAPACKE_?lange itself (lapacke.h arrives with the calaman::lapack_reference
# target). Guard on it and return early when absent: a missing oracle is a
# missing TIER named in the configure output, not a silent pass
# (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping lange tests")
  return()
endif()

# The test/shared/ modules back the device buffers that stage A and the result
# scalar; lange itself takes a bare wwrStream_t and allocates only its scratch.
calaman_add_gtest_executable(
  NAME
  lange_tests
  SOURCES
  lange_tests.cpp
  LINK
  calaman::lange
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::complex
  wwr::extension::memory_buffer
)

# Every case stages A on the device and runs the reduction kernels -- including
# the empty-matrix case, which writes 0 with a device memset rather than
# returning on the host -- so the whole suite is REQUIRES_GPU (labeled `gpu`,
# excluded by `ctest -LE gpu`).
calaman_add_gtest_suite_tests(
  TARGET
  lange_tests
  SUITES
  LangeOracleTests
  REQUIRES_GPU
)
