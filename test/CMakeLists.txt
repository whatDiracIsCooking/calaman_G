# test/ -- calaman's test tree: reusable GPU test utilities under utils/, and
# the suites that exercise them under test/. Backend-neutral like src/: each
# entry is added once and builds for either vendor. Nothing here is installed,
# and nothing under it links the CPU reference LAPACK except a test that
# conditions on calaman::lapack_reference.
#
# utils/ is added first: a suite under test/ links the utility target it needs,
# which must already be defined.
add_subdirectory(utils)
add_subdirectory(test)
