# Oracle suite for calaman.horner -- matrix polynomial evaluation by Horner's
# method. Horner matrix-polynomial evaluation is not a LAPACK routine, so there
# is no LAPACKE_?horner binding; the oracle is the reference CBLAS gemm
# (cblas_?gemm) accumulating the power series p(A) = sum_k c_k A^k on the host,
# which arrives with the same oracle target as LAPACKE. Guard on
# calaman::lapack_reference and return early when it is absent: a missing oracle
# is a missing TIER named in the configure output, not a silent pass
# (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping horner tests"
  )
  return()
endif()

# The two test/shared/ modules back the device buffers that stage A, the
# coefficients and P; horner itself takes a bare wwrblasHandle_t plus a
# caller-supplied workspace and allocates nothing. tolerance supplies eps<T>()
# and the Frobenius norm the polynomial error bound is built on.
calaman_add_gtest_executable(
  NAME
  horner_tests
  SOURCES
  horner_tests.cpp
  LINK
  calaman.horner
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.shared.tolerance
  calaman::lapack_reference
  wwr.blas
  wwr.runtime_api
  wwr.extension.memory_buffer
)

# Two registrations for one target (the drift guard unions them): the workspace
# query and the argument-checking contract are pure host calls -- the arg checks
# return before horner() ever touches the handle or a device pointer, so they
# take a null handle and host dummy pointers and run on a card-less runner. The
# numerical suites stage A/coeffs/P on the device and run the gemm + diagonal
# kernels, so they are REQUIRES_GPU (labeled `gpu`, excluded by ctest -LE gpu).
calaman_add_gtest_suite_tests(
  TARGET
  horner_tests
  SUITES
  HornerBufferSizeTests
  HornerArgCheckTests
)

calaman_add_gtest_suite_tests(
  TARGET
  horner_tests
  SUITES
  HornerOracleTests
  HornerPaddingTests
  REQUIRES_GPU
)
