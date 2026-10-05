# Oracle suite for calaman.random_unitary -- fill n-by-n with standard normal,
# then expand the Q of that Gaussian matrix. REQUIRES_GPU: every case stages
# work on the device, creates a solver handle, seeds device RNG states and runs
# geqrf + orgqr/ungqr, so `ctest -LE gpu` excludes it (labeled `gpu`).
#
# Like the orthogonalize suite, the oracle is NOT the reference LAPACK -- the
# input is random and the Q unique only up to column sign/phase, so the check is
# the one convention-free invariant of a unitary, Q^H Q = I, plus a
# seed-sensitivity check. Both are plain host matrix math in std::complex, so
# this suite needs no calaman::lapack_reference and no device BLAS, and carries
# NO lapack_reference guard -- it runs wherever a card exists.
#
#   wwr.solver                  the solver handle the routine takes
#   wwr.runtime_api             stream sync for the staging copies
#   wwr.rand                    wwrrandState (the caller's generator states)
#   wwr.complex                 backend-neutral host complex accessors
#   wwr.extension.memory_buffer the host/device buffer wrappers
#   wwr.extension.init_state    seed the RNG states before the fill
#   calaman.random_unitary      the routine under test
#   calaman.test.shared.*       abort policy, device handle, tolerance
calaman_add_gtest_executable(
  NAME
  random_unitary_tests
  SOURCES
  random_unitary_tests.cpp
  LINK
  calaman::random_unitary
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::test::shared::tolerance
  wwr::solver
  wwr::runtime_api
  wwr::rand
  wwr::complex
  wwr::extension::memory_buffer
  wwr::extension::init_state
)

calaman_add_gtest_suite_tests(
  TARGET
  random_unitary_tests
  SUITES
  RandomUnitaryOracleTests
  REQUIRES_GPU
)
