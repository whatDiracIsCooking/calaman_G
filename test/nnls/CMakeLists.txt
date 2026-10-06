# Tests for calaman.nnls -- non-negative least squares (Lawson-Hanson active
# set). REQUIRES_GPU: every case stages A/b/x/work on the device and runs the
# active-set kernels (and geqp3's), so `ctest -LE gpu` excludes it. Unlike the
# factorization suites there is no reference routine -- LAPACKE has no nnls --
# so correctness is pinned by the KKT conditions, exact recovery of a known
# non-negative solution, and the A^T b <= 0 => x = 0 case, all checked on the
# host. Hence NO calaman::lapack_reference guard here. The two test/shared
# modules stage the device buffers and tolerance supplies eps/frobenius_norm;
# nnls takes a wwrblasHandle_t plus a wwrsolverDnHandle_t (the ormqr apply),
# hence wwr.blas and wwr.solver for the test's own handle creation.
calaman_add_gtest_executable(
  NAME
  nnls_tests
  SOURCES
  nnls_tests.cpp
  LINK
  calaman::nnls
  calaman::test::shared::abort_policy
  calaman::test::shared::expect_converged
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::test::shared::tolerance
  wwr::blas
  wwr::solver
  wwr::runtime_api
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  nnls_tests
  SUITES
  NnlsKktTests
  NnlsRecoveryTests
  NnlsZeroTests
  NnlsIterationCapTests
  NnlsArgumentTests
  REQUIRES_GPU
)
