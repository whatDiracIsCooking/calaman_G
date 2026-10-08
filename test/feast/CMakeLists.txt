# Oracle suite for calaman.feast -- the FEAST eigensolver. The natural oracle
# for an eigensolver is the reference LAPACK (LAPACKE_?syevd), which arrives as
# calaman::lapack_reference; guard on that target and skip rather than fail the
# configure where it is absent (it is a WARNING, not an error -- CLAUDE.md). The
# filter-shape and diagonal/empty-interval cases carry their own closed-form
# oracles, but they live in this one guarded binary for a single target.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping feast tests"
  )
  return()
endif()

# The two test/shared/ modules back the device buffers that stage the matrices;
# wwr.blas / wwr.solver supply the handles feast needs, wwr.wrappers.common
# the real_fp/RealToComplexType traits, wwr.wrappers.blas the symm and gemms
# behind the KrylovResolvent tests' operators, wwr.runtime_api the stream/sync,
# test.shared.tolerance the bound on the Laplacian's closed-form eigenvalues.
# feast allocates nothing -- the test supplies the workspace.
calaman_add_gtest_executable(
  NAME
  feast_tests
  SOURCES
  feast_tests.cpp
  LINK
  calaman::feast
  calaman::test::shared::abort_policy
  calaman::test::shared::expect_converged
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  wwr::blas
  wwr::solver
  wwr::runtime_api
  wwr::wrappers::common
  wwr::wrappers::blas
  wwr::extension::memory_buffer
  calaman::lapack_reference
)

# The rational filter and the argument-checking contract are pure host calls --
# feast_rational_filter() and feast()'s rejection paths return before any
# device work -- so they run on a card-less runner.
calaman_add_gtest_suite_tests(
  TARGET
  feast_tests
  SUITES
  FeastQuadratureTests
  FeastArgCheckTests
)

# The numerical suites stage the matrices on the device and run the contour
# filter, the batched LU and the Rayleigh-Ritz, so they are REQUIRES_GPU
# (`gpu`, excluded by ctest -LE gpu).
calaman_add_gtest_suite_tests(
  TARGET
  feast_tests
  SUITES
  FeastDiagonalTests
  FeastReferenceTests
  FeastEmptyIntervalTests
  FeastHiddenNormTests
  FeastKrylovResolventTests
  FeastMatrixFreeTests
  REQUIRES_GPU
)
