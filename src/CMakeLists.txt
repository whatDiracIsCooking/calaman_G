# src/ -- calaman's shipped modules. Backend-neutral by construction: there is
# no per-backend subtree (docs/architecture.md §1). Each module is one directory
# written once against WarpWraps's `wwr*` names, added once here, and built for
# whichever backend CALAMAN_GPU_BACKEND selected.
#
# A module that computes what a LAPACK routine computes carries LAPACK's name
# (docs/architecture.md §4).

add_subdirectory(common)
add_subdirectory(diff_norm)
add_subdirectory(gebal)
add_subdirectory(horner)
add_subdirectory(lacpy)
add_subdirectory(orthogonalize)

# The dense linear-algebra routines -- the geqp3 column-pivoted-QR call graph,
# one directory and one module per LAPACK routine. Added in call-graph order so
# each module's dependencies are declared before it: larfg/larf, then laqp2
# (larfg+larf) and laqps (larfg), then the geqp3 driver (laqp2+laqps).
add_subdirectory(larfg)
add_subdirectory(larf)
add_subdirectory(laqp2)
add_subdirectory(laqps)
add_subdirectory(geqp3)
