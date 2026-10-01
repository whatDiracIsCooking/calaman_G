# Scaffold suite for calaman.linalg. Unlike the numerical suites that follow, it
# is host-only (no device memory, no kernel), so it is NOT marked REQUIRES_GPU
# and runs on a card-less CI runner.
#
# It still compares against the reference LAPACK (LAPACKE_dlange as a sanity
# oracle), which arrives as calaman::lapack_reference. Guard on that target and
# return early when it is absent: a missing oracle is a missing TIER named in
# the configure output, not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping linalg tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  linalg_scaffold_tests
  SOURCES
  linalg_scaffold_tests.cpp
  LINK
  calaman.linalg
  calaman.test.shared.tolerance
  calaman::lapack_reference
)

calaman_add_gtest_suite_tests(
  TARGET
  linalg_scaffold_tests
  SUITES
  LinalgScaffoldTests
)

# Oracle suite for calaman.linalg:larfg -- the elementary Householder reflector.
# Unlike the scaffold, it stages [alpha; x] on the device and runs the BLAS
# routines, so it is REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
# It compares against LAPACKE_?larfg; the two test/shared/ modules back the
# device buffers that stage the inputs, and tolerance supplies eps<T>().
calaman_add_gtest_executable(
  NAME
  larfg_tests
  SOURCES
  larfg_tests.cpp
  LINK
  calaman.linalg
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.shared.tolerance
  calaman::lapack_reference
  wwr.blas
  wwr.runtime_api
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  larfg_tests
  SUITES
  LarfgOracleTests
  REQUIRES_GPU
)

# Oracle test for calaman.linalg's larf (applying an elementary reflector).
# REQUIRES_GPU: every case stages C and v on the device and runs the gemv/ger
# kernels, so `ctest -LE gpu` excludes it. It compares against LAPACKE_?larf
# (calaman::lapack_reference), with wwr.extension.memory_buffer + the two
# test/shared/ modules staging the device buffers -- larf itself takes a bare
# wwrblasHandle_t plus a caller-supplied workspace and allocates nothing.
calaman_add_gtest_executable(
  NAME
  larf_tests
  SOURCES
  larf_tests.cpp
  LINK
  calaman.linalg
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.shared.tolerance
  calaman::lapack_reference
  wwr.blas
  wwr.runtime_api
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  larf_tests
  SUITES
  LarfOracleTests
  REQUIRES_GPU
)
