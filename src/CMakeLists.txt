# src/ -- calaman's shipped modules. Backend-neutral by construction: there is
# no per-backend subtree (docs/architecture.md §1). Each module is one directory
# written once against WarpWraps's `wwr*` names, added once here, and built for
# whichever backend CALAMAN_GPU_BACKEND selected.
#
# A module that computes what a LAPACK routine computes carries LAPACK's name
# (docs/architecture.md §4).

add_subdirectory(common)

# calaman.error_handling is a pure-host value type (Status) that unifies the
# three WarpWraps error domains behind one return type. Dependency-free of the
# rest of src/ -- added next to common as foundational shared infrastructure. No
# consumer links it yet; it is landed on its own (diff_norm's cross-domain path
# is the motivating adopter, migrated separately).
add_subdirectory(error_handling)

# calaman.elem_ops is header-only device infrastructure (per-element scalar
# arithmetic over float/double/complex), consumed by other modules' .cu TUs, so
# it is added early -- before any consumer's device library links its INTERFACE
# target. Like calaman.reduce_columns, it is not a module.
add_subdirectory(elem_ops)

add_subdirectory(complex_cast)

# calaman.lacgv is complex_cast's slimmest sibling (a single per-element functor
# through parallel_for, conjugating via calaman.elem_ops) -- the GPU counterpart
# of LAPACK's ?lacgv. It links calaman.elem_ops, added above, so it follows it.
add_subdirectory(lacgv)

# calaman.set_element is complex_cast's structural twin (per-element functors
# through parallel_for, |.| through calaman.elem_ops) -- the device-pointer-mode
# back half of iamax, reading one element of a device array by a device-held
# index. It links calaman.elem_ops, added above, so it follows it.
add_subdirectory(set_element)

add_subdirectory(diff_norm)
add_subdirectory(gebal)
add_subdirectory(horner)

# calaman.paterson_stockmeyer is the asymptotically-cheaper twin of
# calaman.horner (O(sqrt(d)) gemms vs d). It shares no build edge with horner --
# added next to it only for readability.
add_subdirectory(paterson_stockmeyer)

add_subdirectory(lacpy)

# calaman.laset is lacpy's fill twin (set to constants rather than copy). It
# shares no build edge with lacpy -- added next to it only for readability.
add_subdirectory(laset)

# calaman.lascl2 is laset's structural twin (a hand-launched per-element kernel
# over a column-major matrix), here diagonal row-scaling x <-- D * x rather than
# a constant fill. It shares no build edge with laset -- added next to it only
# for readability.
add_subdirectory(lascl2)

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

# calaman.pstf2 -- the unblocked level-2 pivoted Cholesky panel of a symmetric
# positive semidefinite matrix. A host composition of the wrapped BLAS plus its
# own fused diagonal-and-pivot reduction; it shares no build edge with the geqp3
# call graph above (complete pivoting, not column pivoting) and links only
# calaman.common (Uplo) and calaman.error_handling, both added earlier.
add_subdirectory(pstf2)

# calaman.nnls -- non-negative least squares (Lawson-Hanson active set). A host
# driver over wrapped BLAS + wwr::ormqr that solves each passive-set subproblem
# with calaman.geqp3, plus its own P/Z-bookkeeping kernels. Added after geqp3 so
# that target exists when it links it.
add_subdirectory(nnls)

# calaman.expm -- the matrix exponential by scaling and squaring. A host
# composition over wrapped BLAS + the LU solve + calaman.gebal, plus its own
# fused kernels. Added after its calaman dependencies (common, gebal,
# reduce_columns) so their targets exist when it links them.
add_subdirectory(expm)

# calaman.feast -- the FEAST eigensolver for real symmetric matrices. A host
# driver over wrapped batched BLAS + symm/gemm + the eigensolver +
# calaman.orthogonalize, plus its own fused kernels. Added after its calaman
# dependencies (common, orthogonalize) so their targets exist when linked.
add_subdirectory(feast)

# calaman.cg_unitary -- Riemannian conjugate gradient on the unitary group
# U(n). A host driver over wrapped BLAS, calaman.expm (the geodesic step) and
# calaman.orthogonalize (optional reunitarization), plus its own line-search
# kernels. Added after expm and orthogonalize so those targets exist when
# linked.
add_subdirectory(cg_unitary)

# calaman.lasrt -- sort a 1-D array (increasing or decreasing), the GPU
# counterpart of LAPACK's ?lasrt. A single hand-launched device kernel
# (quicksort/insertion), structurally laset's twin but functionally standalone:
# it shares no build edge with the rest of src/ and links only calaman.common
# (SortDir) and calaman.error_handling, both added earlier.
add_subdirectory(lasrt)
