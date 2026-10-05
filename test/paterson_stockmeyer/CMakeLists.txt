# Oracle suite for calaman.paterson_stockmeyer -- matrix polynomial evaluation
# by the Paterson-Stockmeyer scheme. Like calaman.horner this is not a LAPACK
# routine, so there is no LAPACKE_?ps binding; the oracle is the reference CBLAS
# gemm (cblas_?gemm) accumulating the power series p(A) = sum_k c_k A^k on the
# host, which arrives with the same oracle target as LAPACKE. Guard on
# calaman::lapack_reference and return early when it is absent: a missing oracle
# is a missing TIER named in the configure output, not a silent pass
# (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS
    "calaman::lapack_reference not found; skipping paterson_stockmeyer tests"
  )
  return()
endif()

# The two test/shared/ modules back the device buffers that stage A, the
# coefficients and P; paterson_stockmeyer itself takes a bare wwrblasHandle_t
# plus a caller-supplied workspace and allocates nothing. tolerance supplies
# eps<T>() and the Frobenius norm the polynomial error bound is built on.
# calaman.horner is linked for the MatchesHorner cross-check.
calaman_add_gtest_executable(
  NAME
  paterson_stockmeyer_tests
  SOURCES
  paterson_stockmeyer_tests.cpp
  LINK
  calaman::paterson_stockmeyer
  calaman::horner
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  calaman::test::shared::tolerance
  calaman::lapack_reference
  wwr::blas
  wwr::runtime_api
  wwr::extension::memory_buffer
)

# Two registrations for one target (the drift guard unions them): the block-size
# logic, the workspace query and the argument-checking contract are pure host
# calls -- the plan helpers are constexpr and the arg checks return before
# paterson_stockmeyer() ever touches the handle or a device pointer, so they run
# on a card-less runner. The numerical suites stage A/coeffs/P on the device and
# run the gemm + block kernels, so they are REQUIRES_GPU (labeled `gpu`,
# excluded by ctest -LE gpu).
calaman_add_gtest_suite_tests(
  TARGET
  paterson_stockmeyer_tests
  SUITES
  PatersonStockmeyerPlanTests
  PatersonStockmeyerBufferSizeTests
  PatersonStockmeyerArgCheckTests
)

calaman_add_gtest_suite_tests(
  TARGET
  paterson_stockmeyer_tests
  SUITES
  PatersonStockmeyerOracleTests
  PatersonStockmeyerPaddingTests
  REQUIRES_GPU
)
