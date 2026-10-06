# Suite for calaman.davidson. The Euclidean solver's natural oracle is the
# reference LAPACK (LAPACKE_?syevd), which arrives as calaman::lapack_reference;
# guard on that target and skip rather than fail the configure where absent
# (a WARNING, not an error -- CLAUDE.md). The argument-checking and workspace
# cases live in the same guarded binary for a single target, as feast does.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping davidson tests")
  return()
endif()

# The two test/shared/ modules back the device buffers that stage the operator
# and guess; wwr.blas / wwr.solver supply the handles, wwr.wrappers.blas the
# gemv the sigma callback uses, wwr.runtime_api the stream/memcpy/sync.
calaman_add_gtest_executable(
  NAME
  davidson_tests
  SOURCES
  davidson_tests.cpp
  LINK
  calaman::davidson
  calaman::test::shared::abort_policy
  calaman::test::shared::expect_converged
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  wwr::blas
  wwr::solver
  wwr::runtime_api
  wwr::wrappers::common
  wwr::wrappers::blas
  wwr::extension::memory_buffer
  calaman::lapack_reference
)

# The argument-checking contract is pure host calls -- davidson_bufferSize /
# make_davidson_slices / davidson_solve reject a bad shape, a null out-pointer,
# a bad guess_count or a metric callback before any handle use -- so it runs on
# a card-less runner.
calaman_add_gtest_suite_tests(
  TARGET
  davidson_tests
  SUITES
  DavidsonArgCheckTests
)

# Sizing/carving creates a cuSOLVER handle and allocates the workspace; the
# reference suite stages the operator and runs the full iteration on the device.
# Both are REQUIRES_GPU (`gpu`, excluded by ctest -LE gpu).
calaman_add_gtest_suite_tests(
  TARGET
  davidson_tests
  SUITES
  DavidsonBufferSizeTests
  DavidsonReferenceTests
  DavidsonMetricReferenceTests
  REQUIRES_GPU
)
