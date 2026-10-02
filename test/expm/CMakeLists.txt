# Oracle suite for calaman.expm -- the matrix exponential by scaling and
# squaring. There is no LAPACKE_?expm (LAPACK ships no matrix exponential), so
# the oracle is closed forms, identities any exponential satisfies, and an
# independent pade(13) on the same matrix -- the spec is its own oracle.
# So, like the gebal spec suite and unlike the linalg oracle suites, there is NO
# calaman::lapack_reference guard here.
#
# The two test/shared/ modules back the device buffers that stage the matrices;
# wwr.complex and wwr.wrappers.common supply the complex element types and the
# ComplexToRealType the norm is spelled with, wwr.solver the LU handle expm
# needs. expm itself allocates nothing -- the test supplies the workspace.
calaman_add_gtest_executable(
  NAME
  expm_tests
  SOURCES
  expm_tests.cpp
  LINK
  calaman.expm
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.utils.shared_device
  wwr.blas
  wwr.solver
  wwr.runtime_api
  wwr.complex
  wwr.wrappers.common
  wwr.extension.memory_buffer
)

# The plan ladder, the thresholds and the argument-checking contract are pure
# host calls -- expm()/expm_plan()/pade_theta() and the workspace queries return
# before any device work on those paths -- so they run on a card-less runner.
calaman_add_gtest_suite_tests(
  TARGET
  expm_tests
  SUITES
  ExpmPlanTests
  ExpmThetaTests
  ExpmArgCheckTests
)

# The numerical suites stage the matrices on the device and run the Pade
# kernels, the BLAS products and the LU solve, so they are REQUIRES_GPU (`gpu`,
# excluded by ctest -LE gpu).
calaman_add_gtest_suite_tests(
  TARGET
  expm_tests
  SUITES
  ExpmClosedFormTests
  ExpmIdentityTests
  ExpmLadderTests
  ExpmPaddingTests
  ExpmPointerModeTests
  REQUIRES_GPU
)
