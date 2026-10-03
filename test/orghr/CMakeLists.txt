# Oracle suite for calaman.orghr -- generate the orthogonal Q from ?gehrd's
# reflectors. It stages the panel on the device and runs the shuffle kernel plus
# wwr::orgqr, so it is REQUIRES_GPU (labeled `gpu`; `ctest -LE gpu` skips it).
# It compares against reference ?gehrd / ?orghr via their LAPACKE bindings,
# which arrive as calaman::lapack_reference; guard on that target and return
# early when absent, so a missing oracle is a missing tier named in the
# configure output, not a silent pass (docs/architecture.md §3). The three
# test/shared/ modules back the device buffers and supply eps<T>() for the tol.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping orghr tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  orghr_tests
  SOURCES
  orghr_tests.cpp
  LINK
  calaman.orghr
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.shared.tolerance
  calaman.test.utils.shared_device
  calaman::lapack_reference
  wwr.solver
  wwr.runtime_api
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  orghr_tests
  SUITES
  OrghrOracleTests
  REQUIRES_GPU
)
