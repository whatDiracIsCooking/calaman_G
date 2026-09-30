# test/ -- calaman's test tree: support modules shared across targets under
# shared/, reusable GPU test utilities under utils/, and the suites that
# exercise them under test/. Backend-neutral like src/: each entry is added
# once and builds for either vendor. Nothing here is installed, and nothing
# under it links the CPU reference LAPACK except a test that conditions on
# calaman::lapack_reference.
#
# shared/ then utils/ then test/, in dependency order: a utility links the
# shared support modules, and a suite under test/ links the utility target it
# needs -- each must already be defined when the next directory is added.
add_subdirectory(shared)
add_subdirectory(utils)
add_subdirectory(test)
