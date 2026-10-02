# Oracle suite for calaman.orthogonalize -- QR-based in-place orthonormalizer.
# REQUIRES_GPU: every case stages A/work on the device, creates a solver handle
# and runs geqrf + orgqr/ungqr, so `ctest -LE gpu` excludes it (labeled `gpu`).
#
# Unlike the linalg suites, the oracle is NOT the reference LAPACK -- a QR is
# unique only up to column sign/phase, so the checks are the two sign-free
# invariants of an orthonormal basis for range(A): Q^H Q = I, Q(Q^H A) = A.
# They are plain matrix products evaluated on the host in std::complex, so this
# suite needs no calaman::lapack_reference and no device BLAS, and therefore
# carries NO lapack_reference guard -- it runs wherever a card exists.
#
#   wwr.solver                  the solver handle the routine takes
#   wwr.complex                 backend-neutral host complex construct/accessors
#   wwr.runtime_api             stream sync for the staging copies
#   wwr.extension.memory_buffer the host/device buffer wrappers
#   calaman.orthogonalize       the routine under test
#   calaman.test.shared.*       abort policy, device handle, tolerance
calaman_add_gtest_executable(
  NAME
  orthogonalize_tests
  SOURCES
  orthogonalize_tests.cpp
  LINK
  calaman.orthogonalize
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.shared.tolerance
  wwr.solver
  wwr.complex
  wwr.runtime_api
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  orthogonalize_tests
  SUITES
  OrthogonalizeOracleTests
  REQUIRES_GPU
)
