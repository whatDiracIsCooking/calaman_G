# Oracle suite for calaman.laset (built under src/laset/) -- set all or part of
# a matrix to constants. The oracle is the reference LAPACKE_?laset run over the
# identical inputs; ?laset writes only constants, so the device and reference
# agree bit for bit and the suite compares with `==`.
#
# It cannot be built without the oracle, so guard on calaman::lapack_reference
# and return early when it is absent: a missing oracle is a missing TIER named
# in the configure output, not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping laset tests")
  return()
endif()

# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case stages
# the matrix on the device and runs the fill kernel. wwr.extension.memory_buffer
# and the two test/shared/ modules back the device buffers that stage inputs;
# laset itself takes a bare wwrStream_t and allocates nothing.
calaman_add_gtest_executable(
  NAME
  laset_tests
  SOURCES
  laset_tests.cpp
  LINK
  calaman::laset
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  laset_tests
  SUITES
  LasetOracleTests
  REQUIRES_GPU
)
