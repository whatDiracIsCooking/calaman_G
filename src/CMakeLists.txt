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

# calaman.gebak is gebal's back-transformation companion (undo the balancing
# similarity on a matrix of eigenvectors). It shares no build edge with gebal --
# it consumes gebal's scale/ilo/ihi at the call site, not at link -- so it is
# added next to it only for readability.
add_subdirectory(gebak)

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

# calaman.lascl is lascl2's near-namesake and structural twin (the same
# hand-launched per-element kernel), here LAPACK's ?lascl: scale a full matrix
# by cto/cfrom via a host-side over/underflow-guarded multiplier chain rather
# than a diagonal product. It shares no build edge with lascl2 -- added next to
# it only for readability.
add_subdirectory(lascl)

# calaman.lartg is lacgv's structural sibling (a single per-element functor
# through wwr.extension.parallel_for), here the batched lift of LAPACK's ?lartg:
# n plane rotations (c, s, r) from n pairs (f, g). It links only
# wwr.extension.parallel_for (its math goes through wwr.wrappers.math, no
# elem_ops), so it shares no build edge with the rest of src/ -- added next to
# the per-element family only for readability.
add_subdirectory(lartg)

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

# calaman.lange -- the norm of a general matrix (LAPACK's ?lange), as a
# two-stage reduction. Added after calaman.reduce_columns (its reduction engine,
# which both stages defer to) so that target exists when its device lib uses it.
add_subdirectory(lange)

# The dense linear-algebra routines -- the geqp3 column-pivoted-QR call graph,
# one directory and one module per LAPACK routine. Added in call-graph order so
# each module's dependencies are declared before it: larfg/larf, then laqp2
# (larfg+larf) and laqps (larfg), then the geqp3 driver (laqp2+laqps).
add_subdirectory(larfg)
add_subdirectory(larf)

# calaman.larf1f is calaman.larf's sibling: the same reflector applied with an
# implicit unit v(1) (the form ?larfg emits). It shares no build edge with larf
# -- added next to it only for readability.
add_subdirectory(larf1f)

# calaman.larfx is calaman.larf's optimised twin (LAPACK ?larfx): a fused
# workspace-free kernel for reflector order < 11, deferring to calaman.larf for
# order >= 11. It LINKS calaman.larf (the fallback), so it follows it here so
# that target exists when larfx links it.
add_subdirectory(larfx)

add_subdirectory(laqp2)
add_subdirectory(laqps)
add_subdirectory(geqp3)

# calaman.larft -- the triangular factor T of a block reflector (LAPACK ?larft),
# the piece larfb needs to apply a whole Householder block at once. A recursive
# host composition of the wrapped BLAS-3 (trmm/gemm) plus geam; it shares no
# build edge with the geqp3 call graph above and links only calaman.common
# (Direct/StoreV) and calaman.error_handling, both added earlier.
add_subdirectory(larft)

# calaman.larfb -- apply a block reflector H (or H^T) to a general matrix
# (LAPACK ?larfb), the level-3 companion that consumes the factor larft forms. A
# host composition of the wrapped BLAS-3 (trmm/gemm) plus geam for the block
# copies and the rank-k subtraction; it takes T as input rather than calling
# larft, so it shares no build edge with it and links only calaman.common
# (Side/Trans/Direct/StoreV) and calaman.error_handling, both added earlier.
add_subdirectory(larfb)

# calaman.lahr2 -- LAPACK's ?lahr2, reduce the first nb columns of a panel to
# upper Hessenberg form and return the V/T/Y that ?gehrd applies to the trailing
# matrix. A host composition of wrapped BLAS plus calaman.larfg per column
# (structurally calaman.larft's sibling, no custom kernel); it links
# calaman.larfg (added above in the geqp3 call graph) so it follows it.
add_subdirectory(lahr2)

# calaman.pstf2 -- the unblocked level-2 pivoted Cholesky panel of a symmetric
# positive semidefinite matrix. A host composition of the wrapped BLAS plus its
# own fused diagonal-and-pivot reduction; it shares no build edge with the geqp3
# call graph above (complete pivoting, not column pivoting) and links only
# calaman.common (Uplo) and calaman.error_handling, both added earlier.
add_subdirectory(pstf2)

# calaman.pstrf -- the blocked (level-3) pivoted Cholesky driver. Inlines the
# panel look-ahead over the wrapped BLAS (syrk trailing update) and falls back
# to calaman.pstf2 for small n or a degenerate block size; added after pstf2 so
# that target exists when it links it.
add_subdirectory(pstrf)

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

# calaman.lanv2 -- Schur-standardise a real 2-by-2 block (LAPACK's ?lanv2), the
# 2-by-2 deflation step of the real Schur reduction. Pure host scalar arithmetic
# with no BLAS: structurally calaman.larfg's slimmest sibling (an exported
# header-only template) but taking only a stream, since it owns no device code.
# It links only wwr.runtime_api and calaman.error_handling (added earlier).
add_subdirectory(lanv2)

# calaman.ladiv -- real-arithmetic complex division (LAPACK's ?ladiv), the
# device scalar helper laln2 calls per-thread. The scalar routine is header-only
# and host/device; the module batches n of them on the device (calaman.lartg's
# sibling) so the oracle can exercise it on a card. Links only wwr.runtime_api
# and calaman.error_handling (added earlier).
add_subdirectory(ladiv)

# calaman.lasy2 -- solve the order-(1..2) Sylvester equation
# op(TL)*X + ISGN*X*op(TR) = SCALE*B, the GPU counterpart of LAPACK's ?lasy2. A
# single single-thread device kernel (the whole solve is O(1), X at most 2x2),
# structurally lacgv's twin through wwr.extension.parallel_for. It shares no
# build edge with the rest of src/ and links only calaman.error_handling.
add_subdirectory(lasy2)

# calaman.laqr1 -- LAPACK's ?laqr1, the scaled first column of the shift
# polynomial (H-s1*I)(H-s2*I) that ?laqr5 uses to seed each bulge. Pure host
# C++23 (O(1) scalar arithmetic on the top-left 3-by-3 of H), lanv2's sibling:
# it shares no build edge with the rest of src/ and links only
# calaman.error_handling.
add_subdirectory(laqr1)

# calaman.gehd2 -- the unblocked reduction of a general matrix to upper
# Hessenberg form (LAPACK's ?gehd2), the step between calaman.gebal and the QR
# iteration in the nonsymmetric eigenproblem. A host composition and the slim
# sibling of calaman.laqp2: it LINKS calaman.larfg and calaman.larf1f (both
# added earlier in the reflector group), so it follows them here.
add_subdirectory(gehd2)

# calaman.laexc -- swap two adjacent 1x1 / 2x2 diagonal blocks of a real Schur
# form by an orthogonal similarity (LAPACK's ?laexc), a Schur-reordering step.
# One single-thread device kernel (the block math is O(1); the resulting
# rotations/reflectors touch O(N) rows/cols) that inlines the ?lartg/?lasy2/
# ?lanv2/?larfg auxiliaries, so it links no sibling LAPACK module -- only
# calaman.error_handling. Structurally lasy2's twin.
add_subdirectory(laexc)

# calaman.lahqr -- double-shift Francis QR for the Schur form of a small upper
# Hessenberg matrix (LAPACK's ?lahqr), the QR-iteration base case the real
# nonsymmetric eigensolver recurses down to. One single-thread device kernel
# (the whole iteration -- deflation search, Francis shift, bulge-chasing
# reflector sweep, ?lanv2 standardisation -- runs in one thread) that inlines
# the ?larfg/?lanv2/DROT auxiliaries, so it links no sibling LAPACK module --
# only calaman.error_handling. Structurally laexc's twin.
add_subdirectory(lahqr)

# calaman.gehrd -- the blocked reduction of a general matrix to upper Hessenberg
# form (LAPACK's ?gehrd), the level-3 driver over the unblocked calaman.gehd2. A
# host composition that LINKS calaman.lahr2 (per-panel V/T/Y), calaman.larfb
# (the left block-reflector apply) and calaman.gehd2 (the unblocked tail), so it
# follows all three here.
add_subdirectory(gehrd)

# calaman.laln2 -- solve the scaled 1x1 / 2x2 real system
# (ca*op(A) - w*D) X = SCALE*B with a real or complex right-hand side, the GPU
# counterpart of LAPACK's ?laln2 (a ?trevc eigenvector step). One single-thread
# device kernel (the solve is O(1), X at most 2x2), structurally lasy2's twin;
# its complex divisions reach calaman.ladiv's header-only host/device
# ladiv_scalar root-relative (#94 depends on #86), so it links no sibling module
# -- only calaman.error_handling.
add_subdirectory(laln2)

# calaman.laqr5 -- one non-accumulated (KACC22 = 0) multishift QR sweep with
# bulge chasing (LAPACK's ?laqr5), the inner engine of the multishift QR
# iteration. One single-thread device kernel (per-reflector math is O(1); each
# order-2/3 reflector touches O(N) rows/cols) that inlines the ?laqr1 shift
# polynomial and ?larfg reflector generation -- calaman.laqr1 is a host module
# unusable from a device kernel -- so, like laexc, it links no sibling LAPACK
# module, only calaman.error_handling. Placed after laqr1, whose arithmetic it
# reproduces. Structurally laexc's twin.
add_subdirectory(laqr5)

# calaman.ormhr -- multiply a general matrix by the orthogonal Q from ?gehrd
# (LAPACK's ?ormhr: Q*C, Q^T*C, C*Q, C*Q^T), the back-transform step of the
# nonsymmetric eigenproblem. A pure host index shift over wwr::ormqr -- the
# gehrd reflectors are geqrf's layout shifted one row, so ormhr forwards them to
# one ormqr over the shifted C submatrix. Real only (float/double); no .cu of
# its own (ormqr carries the device work). Links calaman.common (Side/Trans) and
# calaman.error_handling; see its CMakeLists.txt for the wwr edges.
add_subdirectory(ormhr)

# calaman.orghr -- generate the orthogonal Q from ?gehrd's Householder
# reflectors (LAPACK's ?orghr, real s/d), the step that turns gehd2's packed
# output into an explicit Q for the nonsymmetric eigenproblem. A host driver
# that shuffles the reflectors one column right with its own hand-launched
# kernel (laset's layout) and hands the window block to wwr::orgqr; it owns no
# sibling-LAPACK build edge and links only calaman.common and
# calaman.error_handling (both added earlier).
add_subdirectory(orghr)

# calaman.trexc -- reorder a real Schur factorization by moving a selected
# diagonal block to another position (LAPACK's ?trexc), a Schur-reordering
# driver. A host composition and the slim sibling of calaman.gehd2: a host loop
# bubbles the block one adjacent swap at a time, so it LINKS calaman.laexc (the
# swap, added just above) and must follow it here.
add_subdirectory(trexc)

# calaman.trevc3 -- the left/right eigenvectors of a real quasi-triangular
# (Schur) matrix by back-substitution (LAPACK's ?trevc3, HOWMNY = 'A'), the
# eigenvector step of the real nonsymmetric eigensolver. One single-thread
# device kernel (the per-block scaled solve is O(1); each back-substitution
# update touches O(N) entries) that inlines the ?laln2 solve -- calaman.laln2
# is a launch functor unusable from a device kernel -- so, like laexc and
# laqr5, it links no sibling LAPACK module, only calaman.error_handling. Its
# inlined solve reaches calaman.ladiv's header-only ladiv_scalar root-relative,
# as laln2 does. Structurally laqr5's twin; placed after laln2, whose
# arithmetic it reproduces.
add_subdirectory(trevc3)

# calaman.laqr2 -- aggressive early deflation for the real nonsymmetric
# eigensolver (LAPACK's ?laqr2, the small-window variant using ?lahqr for the
# window Schur form, not the recursive ?laqr4 of ?laqr3). One single-thread
# device kernel (the window Schur form, the spike-tip deflation, the ?trexc
# reorder, the ?gehrd re-Hessenberg and the slab updates all run in one thread)
# that inlines its ?lahqr/?trexc/?laexc/?lanv2/?larfg/?gehrd/?ormhr auxiliaries
# -- those siblings are host compositions unusable from a device kernel -- so,
# like laqr5, it links no sibling LAPACK module, only calaman.error_handling.
# Structurally laqr5's twin.
add_subdirectory(laqr2)

# calaman.laqr4 -- the small-bulge multishift QR driver with aggressive early
# deflation (LAPACK's ?laqr4). A host composition (calaman.gehrd's shape): the
# driver loop runs on the host and stages the O(1) scalars it inspects through
# device memcpies, launching the calaman.laqr2 (AED), calaman.lahqr (tiny window
# + trailing-submatrix shifts) and calaman.laqr5 (multishift sweep) kernels.
# ?laqr2 -- not the recursive ?laqr3 -- bounds the recursion. Placed last: it
# links lahqr, laqr2 and laqr5, so all three must already be defined.
add_subdirectory(laqr4)

# calaman.laqr3 -- aggressive early deflation (LAPACK's ?laqr3, the recursive
# variant whose window Schur form is the multishift ?laqr4 when JW > NMIN = 75,
# else ?lahqr). A host composition: because ?laqr4 is a host loop no device
# kernel can call, it splits ?laqr2's single AED kernel into laqr3_setup /
# laqr3_finish halves with calaman.laqr4 driven between them on the host, and
# delegates the JW <= NMIN regime (where ?laqr3 == ?laqr2) whole to
# calaman.laqr2. Placed last: it links both laqr2 and laqr4, so both must
# already be defined.
add_subdirectory(laqr3)

# calaman.laqr0 -- the small-bulge multishift QR driver with aggressive early
# deflation for the Schur form of a Hessenberg matrix (LAPACK's ?laqr0). A host
# composition that reuses calaman.laqr4's driver verbatim, swapping only two
# kernels: the recursive ?laqr3 for AED (where ?laqr4 uses ?laqr2) and ?laqr4
# for the trailing-submatrix shift recomputation (where ?laqr4 uses ?lahqr).
# Placed last: it links lahqr, laqr3, laqr4 and laqr5, so all four must already
# be defined.
add_subdirectory(laqr0)

# calaman.hseqr -- the ?hseqr driver: eigenvalues or the full Schur form of an
# upper Hessenberg matrix, dispatching calaman.laqr0 (n > NMIN = 75) vs
# calaman.lahqr (n <= NMIN) at the ILAENV(12) crossover. A host composition that
# links both dispatch targets plus calaman.laset (Z init, trash clear), so it is
# placed after laqr0 (and therefore after lahqr/laset) once all are defined.
add_subdirectory(hseqr)
