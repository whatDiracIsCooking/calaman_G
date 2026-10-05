# Scaffold suite for the linalg tier. Unlike the numerical oracle suites, it is
# host-only (no device memory, no kernel), so it is NOT marked REQUIRES_GPU and
# runs on a card-less CI runner.
#
# It still compares against the reference LAPACK (LAPACKE_dlange as a sanity
# oracle), which arrives as calaman::lapack_reference. Guard on that target and
# return early when it is absent: a missing oracle is a missing TIER named in
# the configure output, not a silent pass (docs/architecture.md §3).
if(NOT TARGET calaman::lapack_reference)
  message(
    STATUS "calaman::lapack_reference not found; skipping linalg scaffold tests"
  )
  return()
endif()

calaman_add_gtest_executable(
  NAME
  linalg_scaffold_tests
  SOURCES
  linalg_scaffold_tests.cpp
  LINK
  calaman::larfg
  calaman::larf
  calaman::laqp2
  calaman::laqps
  calaman::geqp3
  calaman::test::shared::tolerance
  calaman::lapack_reference
)

calaman_add_gtest_suite_tests(
  TARGET
  linalg_scaffold_tests
  SUITES
  LinalgScaffoldTests
)
