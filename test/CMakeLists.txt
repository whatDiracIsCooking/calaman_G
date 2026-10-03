# test/ -- calaman's test tree: support modules shared across targets under
# shared/, reusable GPU test utilities under utils/, and the suites that
# exercise them, one directory per suite. Backend-neutral like src/: each entry
# is added once and builds for either vendor. Nothing here is installed, and
# nothing under it links the CPU reference LAPACK except a test that conditions
# on calaman::lapack_reference.
#
# shared/ then utils/ then the suites, in dependency order: a utility links the
# shared support modules, and a suite links the utility target it needs -- each
# must already be defined when the next directory is added.
add_subdirectory(shared)
add_subdirectory(utils)

add_subdirectory(cg_unitary)
add_subdirectory(columnwise_ell1)
add_subdirectory(columnwise_ell2)
add_subdirectory(complex_cast)
add_subdirectory(diff_norm)
add_subdirectory(error_handling)
add_subdirectory(expm)
add_subdirectory(feast)
add_subdirectory(gebal)
add_subdirectory(geqp3)
add_subdirectory(horner)
add_subdirectory(lacgv)
add_subdirectory(laqp2)
add_subdirectory(laqps)
add_subdirectory(larf)
add_subdirectory(larfg)
add_subdirectory(lascl2)
add_subdirectory(laset)
add_subdirectory(lasrt)
add_subdirectory(nnls)
add_subdirectory(linalg_scaffold)
add_subdirectory(orthogonalize)
add_subdirectory(paterson_stockmeyer)
add_subdirectory(pstf2)
add_subdirectory(random_unitary)
add_subdirectory(set_element)
