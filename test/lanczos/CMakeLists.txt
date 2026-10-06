# Suite for calaman.lanczos: the argument-checking contract, the workspace
# sizing/carving, and the device kernels behind lanczos_bridge.h. The
# reference-LAPACK suites (and the guard on calaman::lapack_reference they
# need) arrive with the solve.
#
# lanczos_kernel_tests.cpp #includes "lanczos/lanczos_bridge.h" to reach the
# launchers: the src/ root comes through calaman::common, and wwr::backend
# supplies the backend define runtime.h behind it reads.
calaman_add_gtest_executable(
  NAME
  lanczos_tests
  SOURCES
  lanczos_tests.cpp
  lanczos_kernel_tests.cpp
  LINK
  calaman::lanczos
  calaman::common
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  wwr::backend
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

# Sizing creates a solver handle (syevd_bufferSize), carving allocates the
# workspace, and the kernel suite launches lanczos.cu against host-computed
# expectations: REQUIRES_GPU (`gpu`, excluded by ctest -LE gpu).
calaman_add_gtest_suite_tests(
  TARGET
  lanczos_tests
  SUITES
  LanczosBufferSizeTests
  LanczosKernelTests
  REQUIRES_GPU
)
