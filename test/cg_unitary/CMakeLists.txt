# Tests for calaman.cg_unitary -- Riemannian conjugate gradient on U(n),
# exercised on the Brockett criterion J(W) = trace{W^H R W N}. The oracle is a
# Hermitian eigendecomposition: maximizing the criterion drives W^H R W to a
# diagonal of R's eigenvalues in ascending order, so the computed diagonal is
# compared against LAPACKE's eigenvalues (and the off-diagonals checked to
# vanish). That reference is calaman::lapack_reference, so -- like the geqp3
# oracle suite -- this guards on it and builds only when it is present.
#
# REQUIRES_GPU: every case stages R/N/W/work on the device and runs the
# line-search kernels + BLAS + expm, so `ctest -LE gpu` excludes it. cg_unitary
# takes a wwrblasHandle_t plus a wwrsolverDnHandle_t (the geodesic step's expm),
# hence wwr.blas and wwr.solver for the test's own handle creation.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping cg_unitary")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  cg_unitary_tests
  SOURCES
  cg_unitary_tests.cpp
  LINK
  calaman::cg_unitary
  calaman::test::shared::abort_policy
  calaman::test::shared::expect_converged
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::test::shared::tolerance
  calaman::lapack_reference
  wwr::blas
  wwr::solver
  wwr::runtime_api
  wwr::complex
  wwr::wrappers::common
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  cg_unitary_tests
  SUITES
  CgUnitaryBrockettTests
  REQUIRES_GPU
)
