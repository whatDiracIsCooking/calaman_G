# Suite for calaman.lanczos: the argument-checking contract, the workspace
# sizing/carving, the device kernels behind lanczos_bridge.h, and the Ritz
# extraction and the solve against the reference LAPACK.
#
# lanczos_kernel_tests.cpp and lanczos_ritz_tests.cpp #include
# "lanczos/lanczos_bridge.h" to reach the launchers: the src/ root comes through
# calaman::common, and wwr::backend supplies the backend define runtime.h behind
# it reads.
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
# matvec is rejected before any handle use, and the Ritz selection is host
# math, so these run on a card-less runner.
calaman_add_gtest_suite_tests(
  TARGET
  lanczos_tests
  SUITES
  LanczosArgCheckTests
  LanczosSelectTests
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

# The Ritz-extraction oracle is LAPACKE_?syevd (plus ?geqrf/?orgqr for the
# orthonormal basis), via calaman::lapack_reference. Missing is a WARNING, not
# an error, so only this binary is skipped where it is absent.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping lanczos_ritz_tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lanczos_ritz_tests
  SOURCES
  lanczos_ritz_tests.cpp
  LINK
  calaman::lanczos
  calaman::common
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::backend
  wwr::blas
  wwr::solver
  wwr::runtime_api
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  lanczos_ritz_tests
  SUITES
  LanczosRitzReferenceTests
  REQUIRES_GPU
)

# The single-cycle solve against LAPACKE_?syevd (plus ?geqrf/?orgqr for the
# random orthogonal Q of A = Q diag(lambda) Q^T), through a symv-backed matvec
# adapter: wwr::wrappers::blas supplies symv. Same lapack_reference guard.
calaman_add_gtest_executable(
  NAME
  lanczos_solve_tests
  SOURCES
  lanczos_solve_tests.cpp
  LINK
  calaman::lanczos
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::blas
  wwr::solver
  wwr::runtime_api
  wwr::wrappers::blas
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  lanczos_solve_tests
  SUITES
  LanczosSolveReferenceTests
  REQUIRES_GPU
)
