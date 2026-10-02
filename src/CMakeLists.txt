# src/ -- calaman's shipped modules. Backend-neutral by construction: there is
# no per-backend subtree (docs/architecture.md §1). Each module is one directory
# written once against WarpWraps's `wwr*` names, added once here, and built for
# whichever backend CALAMAN_GPU_BACKEND selected.
#
# A module that computes what a LAPACK routine computes carries LAPACK's name
# (docs/architecture.md §4).

add_subdirectory(common)
add_subdirectory(complex_cast)
add_subdirectory(diff_norm)
add_subdirectory(gebal)
add_subdirectory(horner)

# calaman.paterson_stockmeyer is the asymptotically-cheaper twin of
# calaman.horner (O(sqrt(d)) gemms vs d). It shares no build edge with horner --
# added next to it only for readability.
add_subdirectory(paterson_stockmeyer)

add_subdirectory(lacpy)
add_subdirectory(orthogonalize)

# calaman.random_unitary composes calaman.orthogonalize with a Gaussian fill, so
# it is added after orthogonalize: that module's target must exist to be linked.
add_subdirectory(random_unitary)

# calaman.reduce_columns is header-only infrastructure consumed by
# calaman.columnwise_ell1 and calaman.columnwise_ell2, so it is added before
# them: its INTERFACE target must exist for their device libraries to link it.
add_subdirectory(reduce_columns)
add_subdirectory(columnwise_ell1)
add_subdirectory(columnwise_ell2)

# The dense linear-algebra routines -- the geqp3 column-pivoted-QR call graph,
# one directory and one module per LAPACK routine. Added in call-graph order so
# each module's dependencies are declared before it: larfg/larf, then laqp2
# (larfg+larf) and laqps (larfg), then the geqp3 driver (laqp2+laqps).
add_subdirectory(larfg)
add_subdirectory(larf)
add_subdirectory(laqp2)
add_subdirectory(laqps)
add_subdirectory(geqp3)

# calaman.expm -- the matrix exponential by scaling and squaring. A host
# composition over wrapped BLAS + the LU solve + calaman.gebal, plus its own
# fused kernels. Added after its calaman dependencies (common, gebal,
# reduce_columns) so their targets exist when it links them.
add_subdirectory(expm)

# calaman.feast -- the FEAST eigensolver for real symmetric matrices. A host
# driver over wrapped batched BLAS + symm/gemm + the eigensolver +
# calaman.orthogonalize, plus its own fused kernels. Added after its calaman
# dependencies (common, orthogonalize) so their targets exist when it links them.
add_subdirectory(feast)
