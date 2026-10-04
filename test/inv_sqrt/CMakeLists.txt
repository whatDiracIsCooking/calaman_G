# Oracle suite for calaman.inv_sqrt -- the symmetric inverse square root
# X = S^{-1/2}. Like calaman.orthogonalize, the oracle is NOT the reference
# LAPACK: the symmetric PD root is unique, so the defining invariants are
# sign-free and plain host matrix products -- X^T S X = I (X recovers S^{-1/2})
# and X = X^T (symmetric by construction). Computed on the host after X is
# copied back, so the suite needs neither device BLAS nor calaman::lapack_reference
# and runs wherever a card exists.
#
# The two test/shared/ modules back the device buffers staging the matrices;
# wwr.blas / wwr.solver supply the handles (inv_sqrt takes both), wwr.runtime_api
# the stream/sync, calaman.test.shared.tolerance the shared factorization bound.
calaman_add_gtest_executable(
  NAME
  inv_sqrt_tests
  SOURCES
  inv_sqrt_tests.cpp
  LINK
  calaman.inv_sqrt
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.utils.shared_device
  calaman.test.shared.tolerance
  wwr.blas
  wwr.solver
  wwr.runtime_api
  wwr.extension.memory_buffer
)

# Every case stages S/scratch on the device, creates the handles and runs
# syevd + the inverse_sqrt kernel + dgmm + gemm, so the suite is REQUIRES_GPU
# (`gpu`, excluded by ctest -LE gpu).
calaman_add_gtest_suite_tests(
  TARGET
  inv_sqrt_tests
  SUITES
  InvSqrtOracleTests
  REQUIRES_GPU
)
