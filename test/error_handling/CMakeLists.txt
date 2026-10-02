# Suite for calaman.error_handling -- the Status value type unifying the three
# WarpWraps error domains. Host-only (no device, no kernel, no oracle), so it is
# NOT marked REQUIRES_GPU and runs on a card-less CI runner. It links the module
# under test plus the three neutral layers for the enum constants it names
# (WWRBLAS_STATUS_*, WWRSOLVER_STATUS_SUCCESS, wwrSuccess/wwrErrorInvalidValue).
calaman_add_gtest_executable(
  NAME
  error_handling_tests
  SOURCES
  error_handling_tests.cpp
  LINK
  calaman.error_handling
  wwr.extension.common
  wwr.blas
  wwr.solver
  wwr.runtime_api
)

calaman_add_gtest_suite_tests(
  TARGET
  error_handling_tests
  SUITES
  ErrorHandlingTests
)
