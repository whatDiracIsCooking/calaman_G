# src/ -- calaman's shipped modules. Backend-neutral by construction: there is
# no per-backend subtree (docs/architecture.md §1). Each module is one directory
# written once against WarpWraps's `wwr*` names, added once here, and built for
# whichever backend CALAMAN_GPU_BACKEND selected.
#
# A module that computes what a LAPACK routine computes carries LAPACK's name
# (docs/architecture.md §4).

add_subdirectory(common)
add_subdirectory(diff_norm)
add_subdirectory(lacpy)
