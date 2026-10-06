# src/ -- calaman's shipped modules. Backend-neutral by construction: there is
# no per-backend subtree (docs/architecture.md §1). Each module is one directory
# written once against WarpWraps's `wwr*` names, added once here, and built for
# whichever backend CALAMAN_GPU_BACKEND selected.
#
# A module that computes what a LAPACK routine computes carries LAPACK's name
# (docs/architecture.md §4), and those modules live together under src/lapack/
# (added below). This level holds the shared utilities they are written against
# (calaman.common -- which also carries the elem_ops.cuh device header --
# calaman.error_handling, calaman.reduce_columns, calaman.diff_norm, ...) and
# the original algorithms that are not named after a LAPACK routine (expm,
# feast, nnls, orthogonalize, cg_unitary, the columnwise reductions, ...).

add_subdirectory(common)

# calaman.error_handling is a pure-host value type (Status) that unifies the
# three WarpWraps error domains behind one return type. Dependency-free of the
# rest of src/ -- added next to common as foundational shared infrastructure. No
# consumer links it yet; it is landed on its own (diff_norm's cross-domain path
# is the motivating adopter, migrated separately).
add_subdirectory(error_handling)

# calaman.iterative -- IterationInfo/stop_reason/converged(), the outcome
# report the iterative methods share. std only, so its placement is free; it
# must precede nnls, feast and cg_unitary, which link it.
add_subdirectory(iterative)

add_subdirectory(complex_cast)

add_subdirectory(set_element)

add_subdirectory(diff_norm)

add_subdirectory(horner)

# calaman.paterson_stockmeyer is the asymptotically-cheaper twin of
# calaman.horner (O(sqrt(d)) gemms vs d). It shares no build edge with horner --
# added next to it only for readability.
add_subdirectory(paterson_stockmeyer)

add_subdirectory(orthogonalize)

# calaman.random_unitary composes calaman.orthogonalize with a Gaussian fill, so
# it is added after orthogonalize: that module's target must exist to be linked.
add_subdirectory(random_unitary)

# calaman.inv_sqrt is the symmetric (Loewdin) cousin of calaman.orthogonalize --
# the eigen-root X = S^{-1/2} of a symmetric PSD matrix. It depends only on
# calaman.common / calaman.error_handling, so its placement is for
# discoverability beside orthogonalize, not a link order.
add_subdirectory(inv_sqrt)

# calaman.reduce_columns is header-only infrastructure consumed by
# calaman.columnwise_ell1 and calaman.columnwise_ell2, so it is added before
# them: its INTERFACE target must exist for their device libraries to link it.
add_subdirectory(reduce_columns)
add_subdirectory(columnwise_ell1)
add_subdirectory(columnwise_ell2)

# The LAPACK-named routines -- one directory and one module per routine --
# collected under src/lapack/. Added here (not earlier) because every module in
# there links the utilities declared above (common -- with elem_ops.cuh --
# error_handling, diff_norm, reduce_columns), and added here (not later) because
# calaman.nnls below links one of them (calaman.geqp3). See
# src/lapack/CMakeLists.txt for the call-graph order the subtree keeps
# internally.
add_subdirectory(lapack)

# calaman.nnls -- non-negative least squares (Lawson-Hanson active set). A host
# driver over wrapped BLAS + wwr::ormqr that solves each passive-set subproblem
# with calaman.geqp3, plus its own P/Z-bookkeeping kernels. Added after
# src/lapack/ so calaman.geqp3's target exists when it links it.
add_subdirectory(nnls)

# calaman.expm -- the matrix exponential by scaling and squaring. A host
# composition over wrapped BLAS + the LU solve, plus its own fused kernels.
# Added after its calaman dependencies (common -- which carries elem_ops.cuh --
# error_handling, reduce_columns) so their targets exist when it links them.
add_subdirectory(expm)

# calaman.feast -- the FEAST eigensolver for real symmetric matrices. A host
# driver over wrapped batched BLAS + symm/gemm + the eigensolver +
# calaman.orthogonalize, plus its own fused kernels. Added after its calaman
# dependencies (common, orthogonalize) so their targets exist when linked.
add_subdirectory(feast)

# calaman.davidson -- block-Davidson eigensolver for the lowest n_roots
# eigenpairs of a symmetric operator given as a matrix-vector-product callback.
# A host driver over wrapped BLAS + the wrapped eigensolver, with no device code
# of its own. Links only common and error_handling (both added at the top), so
# its placement is free; grouped after feast as the other subspace eigensolver.
add_subdirectory(davidson)

# calaman.lanczos -- thick-restart Lanczos for the extreme eigenpairs of a
# symmetric operator given as a single-vector matvec callback. Links only
# common and error_handling from src/, so its placement is free; grouped with
# davidson as the other callback-driven eigensolver.
add_subdirectory(lanczos)

# calaman.diis -- Pulay DIIS extrapolation over a device-resident ring of
# (vector, residual) pairs. Links only common and error_handling from src/, so
# its placement is free; grouped with the iterative-solver helpers.
add_subdirectory(diis)

# calaman.cg_unitary -- Riemannian conjugate gradient on the unitary group
# U(n). A host driver over wrapped BLAS, calaman.expm (the geodesic step) and
# calaman.orthogonalize (optional reunitarization), plus its own line-search
# kernels. Added after expm and orthogonalize so those targets exist when
# linked.
add_subdirectory(cg_unitary)
