# Oracle suite for calaman.lartg (built under src/lartg/) -- the batched device
# lift of ?lartg: n plane rotations (c, s, r) from n pairs (f, g). The oracle is
# the Fortran reference ?lartg itself (slartg_ / dlartg_), which LAPACKE exposes
# no C binding for, so this suite NEEDS calaman::lapack_reference -- it links
# LAPACK::LAPACK, where those symbols live. Guard on the target and return early
# when absent: a missing oracle is a missing tier named in the configure output,
# not a silent pass (see docs/architecture.md).
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping lartg tests"
  )
  return()
endif()

# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case stages
# the inputs on the device and runs the kernel. wwr.extension.memory_buffer and
# the two test/shared/ modules back the device buffers that stage inputs; lartg
# itself takes a bare wwrStream_t and allocates nothing.
calaman_add_gtest_executable(
  NAME
  lartg_tests
  SOURCES
  lartg_tests.cpp
  LINK
  calaman::lartg
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  lartg_tests
  SUITES
  LartgOracleTests
  REQUIRES_GPU
)
