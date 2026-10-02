# Oracle suite for calaman.laqps -- the level-3 blocked pivoted-QR panel.
# REQUIRES_GPU: every case stages A/tau/vn1/vn2 and the laqps scratch
# (F/auxv/flags) on the device and runs the reflector + downdate kernels, so
# `ctest -LE gpu` excludes it. LAPACKE ships no ?laqps binding, so the oracle is
# LAPACKE_?geqp3 over the whole matrix (laqps at offset 0 with nb >= min(m,n)
# factors it as one block), which arrives as calaman::lapack_reference; guard
# on that target and return early when it is absent -- a missing oracle is a
# missing TIER named in the configure output, not a silent pass
# (docs/architecture.md §3). The two test/shared/ modules stage the device
# buffers and tolerance
# supplies the factorization bound. laqps takes a bare wwrblasHandle_t plus
# caller-supplied device arrays and allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping laqps tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  laqps_tests
  SOURCES
  laqps_tests.cpp
  LINK
  calaman.laqps
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.utils.shared_device
  calaman.test.shared.tolerance
  calaman::lapack_reference
  wwr.blas
  wwr.runtime_api
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  laqps_tests
  SUITES
  LaqpsOracleTests
  REQUIRES_GPU
)
