# Oracle test for calaman.hseqr -- the ?hseqr driver: eigenvalues or the full
# Schur form of an upper Hessenberg matrix, dispatching lahqr vs laqr0 at the
# NMIN = 75 crossover. REQUIRES_GPU: every case stages H / Z on the device and
# runs the dispatched QR kernels, so `ctest -LE gpu` excludes it. It compares
# with the reference LAPACK ?hseqr, which arrives as calaman::lapack_reference;
# guard on that target and return early when it is absent -- a missing oracle is
# a missing TIER named in the configure output, not a silent pass
# (docs/architecture.md section 3). Unlike the ?laqr* / ?lahqr auxiliaries,
# ?hseqr HAS a LAPACKE C binding, so the suite uses LAPACKE_shseqr/dhseqr (via
# lapacke.h) rather than a raw Fortran symbol. The two test/shared/ modules and
# wwr.extension.memory_buffer stage the device data; hseqr itself takes a bare
# stream and allocates nothing.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping hseqr tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  hseqr_tests
  SOURCES
  hseqr_tests.cpp
  LINK
  calaman.hseqr
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.shared.tolerance
  calaman.test.utils.shared_device
  calaman::lapack_reference
  wwr.runtime_api
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  hseqr_tests
  SUITES
  HseqrOracleTests
  REQUIRES_GPU
)
