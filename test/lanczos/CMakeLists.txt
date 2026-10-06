# Suite for calaman.lanczos. Skeleton stage: the argument-checking contract and
# the workspace sizing/carving. The reference-LAPACK suites (and the guard on
# calaman::lapack_reference they need) arrive with the solve.
calaman_add_gtest_executable(
  NAME
  lanczos_tests
  SOURCES
  lanczos_tests.cpp
  LINK
  calaman::lanczos
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  wwr::blas
  wwr::solver
  wwr::runtime_api
  wwr::rand
  wwr::extension::memory_buffer
)

# Pure host calls: a bad shape, a null out-pointer, a null result or an empty
# matvec is rejected before any handle use, so this runs on a card-less runner.
calaman_add_gtest_suite_tests(
  TARGET
  lanczos_tests
  SUITES
  LanczosArgCheckTests
)

# Sizing creates a solver handle (syevd_bufferSize) and carving allocates the
# workspace: REQUIRES_GPU (`gpu`, excluded by ctest -LE gpu).
calaman_add_gtest_suite_tests(
  TARGET
  lanczos_tests
  SUITES
  LanczosBufferSizeTests
  REQUIRES_GPU
)
