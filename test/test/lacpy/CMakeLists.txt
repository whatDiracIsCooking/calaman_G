# Oracle test for calaman.lacpy (built under src/lacpy/). REQUIRES_GPU labels
# the suite `gpu`, so `ctest -LE gpu` (the card-less CI presets) excludes it
# -- every case launches the kernel.
#
# The suite compares against LAPACKE_?lacpy, so it cannot be built without the
# oracle. Guard on calaman::lapack_reference and return early when it is absent:
# a missing oracle is a missing TIER named in the configure output, not a silent
# pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping lacpy oracle tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  lacpy_tests
  SOURCES
  lacpy_tests.cpp
  LINK
  calaman.lacpy
  calaman.test.elementwise_compare
  calaman::lapack_reference
  wwr.runtime_api
  wwr.extension.error_handling
  wwr.extension.runtime
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  lacpy_tests
  SUITES
  LacpyOracleTests
  REQUIRES_GPU
)
