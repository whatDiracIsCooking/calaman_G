# src/lapack/ -- the modules that carry a LAPACK routine's name, one directory
# and one module per routine (docs/architecture.md §4). Split out of src/ so the
# LAPACK-shaped work sits apart from the shared utilities (calaman.common --
# which also carries the elem_ops.cuh device header -- calaman.error_handling,
# calaman.reduce_columns, ...) and the original algorithms (expm, feast, nnls,
# orthogonalize, ...) that remain one level up.
#
# Backend-neutral by construction, exactly like the rest of src/: each module is
# written once against WarpWraps's `wwr*` names and built for whichever backend
# CALAMAN_GPU_BACKEND selected. The include root for every root-relative
# cross-module header stays ${PROJECT_SOURCE_DIR}/src (exported by the utility
# modules one level up), so a module here reaches calaman.error_handling's
# "error_handling/error_macros.h", calaman.common's "common/enums.h", and the
# one intra-LAPACK header "lapack/ladiv/ladiv.h" off that same root.
#
# This directory is added from src/CMakeLists.txt AFTER the utilities these
# modules link (common -- with elem_ops.cuh -- error_handling, diff_norm,
# reduce_columns) and
# BEFORE the one utility that links back into here (calaman.nnls, which links
# calaman.geqp3).
# Modules below are added in call-graph order so each one's dependencies are
# declared first -- the same discipline src/CMakeLists.txt keeps.

# calaman.lacgv is complex_cast's slimmest sibling (a single per-element functor
# through parallel_for, conjugating via calaman.common's elem_ops.cuh) -- the
# GPU counterpart of LAPACK's ?lacgv. It links calaman.common (added one level
# up), so it is safe here.
add_subdirectory(lacgv)

add_subdirectory(gebal)

# calaman.gebak is gebal's back-transformation companion (undo the balancing
# similarity on a matrix of eigenvectors). It shares no build edge with gebal --
# it consumes gebal's scale/ilo/ihi at the call site, not at link -- so it is
# added next to it only for readability.
add_subdirectory(gebak)

add_subdirectory(lacpy)

# calaman.lacp2 is lacpy's real-to-complex twin (the same kernel with a
# widening store). It shares no build edge with lacpy -- added next to it only
# for readability.
add_subdirectory(lacp2)

# calaman.lag2 -- general matrix, double <-> single (?lag2?), plus the
# lag2_convert header (calaman::lag2::header) that calaman.lat2 shares, so it
# must precede lat2. Links only calaman.common and calaman.error_handling.
add_subdirectory(lag2)

# calaman.lat2 -- triangular matrix, double to single (?lat2?): lag2's element
# conversion (calaman::lag2::header) over one triangle, so it follows lag2.
add_subdirectory(lat2)

# calaman.laset is lacpy's fill twin (set to constants rather than copy). It
# shares no build edge with lacpy -- added next to it only for readability.
add_subdirectory(laset)

# calaman.lascl2 is laset's structural twin (a hand-launched per-element kernel
# over a column-major matrix), here diagonal row-scaling x <-- D * x rather than
# a constant fill. It shares no build edge with laset -- added next to it only
# for readability.
add_subdirectory(lascl2)

# calaman.larscl2 is lascl2's inverse twin, x <-- D^-1 x, with no build edge
# to it -- added next to it only for readability.
add_subdirectory(larscl2)

# calaman.lascl is lascl2's near-namesake and structural twin (the same
# hand-launched per-element kernel), here LAPACK's ?lascl: scale a full matrix
# by cto/cfrom via a host-side over/underflow-guarded multiplier chain rather
# than a diagonal product. It shares no build edge with lascl2 -- added next to
# it only for readability.
add_subdirectory(lascl)

# calaman.rscl -- LAPACK's ?rscl: scale a strided VECTOR by 1/a through the same
# host-side guarded multiplier chain lascl runs for cto/cfrom (?rscl's loop is
# ?lascl's with cto = 1). Its own kernel is one per-element functor through
# wwr.extension.parallel_for, so it shares no build edge with lascl -- added
# next to it only for readability.
add_subdirectory(rscl)

# calaman.lartg is lacgv's structural sibling (a single per-element functor
# through wwr.extension.parallel_for), here the batched lift of LAPACK's ?lartg:
# n plane rotations (c, s, r) from n pairs (f, g). It links only
# wwr.extension.parallel_for (its math goes through wwr.wrappers.math, no
# elem_ops), so it shares no build edge with the rest of src/ -- added next to
# the per-element family only for readability.
add_subdirectory(lartg)

# calaman.laruv -- LAPACK's uniform (0,1) generator, bit for bit, plus the
# laruv_block header (calaman::laruv::header) that ?larnv calls, so it must
# precede larnv. Links only calaman.common and calaman.error_handling.
add_subdirectory(laruv)

# calaman.larnv -- a random vector from LAPACK's stream (?larnv), over the
# laruv_block header, so it follows laruv.
add_subdirectory(larnv)

# calaman.lange -- the norm of a general matrix (LAPACK's ?lange), as a
# two-stage reduction. calaman.reduce_columns (its reduction engine, which both
# stages defer to) is added one level up, so its target exists when this device
# lib uses it.
add_subdirectory(lange)

# calaman.ilalr / calaman.ilalc -- the last non-zero row / column (LAPACK's
# ila?lr / ila?lc), max-reductions whose stage two is calaman.reduce_columns,
# added one level up like lange's.
add_subdirectory(ilalr)
add_subdirectory(ilalc)

# calaman.langb -- the norm of a general band matrix in band storage (LAPACK's
# ?langb). Links only calaman.common and calaman.error_handling.
add_subdirectory(langb)

# calaman.langt -- the norm of a general tridiagonal (LAPACK's ?langt). Links
# only calaman.common and calaman.error_handling.
add_subdirectory(langt)

# calaman.lanhs -- the norm of an upper Hessenberg matrix (LAPACK's ?lanhs).
# Links only calaman.common and calaman.error_handling.
add_subdirectory(lanhs)

# calaman.lanst -- the norm of a symmetric tridiagonal (LAPACK's ?lanst), plus
# the header-only per-thread lanst_max_abs (calaman::lanst::header) that the
# ?sterf/?steqr kernels include, so it must precede both. Links only
# calaman.common and calaman.error_handling (added one level up).
add_subdirectory(lanst)

# calaman.lanht -- the norm of a Hermitian tridiagonal (LAPACK's ?lanht). No
# device code of its own: it runs calaman.lanst's kernel with a complex
# off-diagonal and links calaman.lanst.device, so it follows lanst.
add_subdirectory(lanht)

# calaman.lansy -- the norm of a symmetric matrix from one triangle (LAPACK's
# ?lansy), which ?syev scales by. Links only calaman.common and
# calaman.error_handling (added one level up).
add_subdirectory(lansy)

# calaman.lanhe -- the norm of a Hermitian matrix from one triangle (LAPACK's
# ?lanhe). No device code of its own: it runs calaman.lansy's kernel with the
# Hermitian flag and links calaman.lansy.device, so it follows lansy.
add_subdirectory(lanhe)

# calaman.lassq -- LAPACK's ?lassq: the over/underflow-safe scaled sum of
# squares of a vector, accumulated into a caller-held (scale, sumsq) pair. The
# accumulator the lan* Frobenius norms above take a plain-sum shortcut over, so
# it is added after them; it shares no build edge with any of them, linking only
# calaman.common and calaman.error_handling (added one level up).
add_subdirectory(lassq)

# calaman.lasr -- apply a sequence of plane rotations (LAPACK's ?lasr), plus the
# block-cooperative lasr_block header (calaman::lasr::header) that the ?steqr
# kernel includes, so it must precede it. Links only calaman.common and
# calaman.error_handling (added one level up).
add_subdirectory(lasr)

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
# (Direct/StoreV) and calaman.error_handling, both added one level up.
add_subdirectory(larft)

# calaman.larfb -- apply a block reflector H (or H^T) to a general matrix
# (LAPACK ?larfb), the level-3 companion that consumes the factor larft forms. A
# host composition of the wrapped BLAS-3 (trmm/gemm) plus geam for the block
# copies and the rank-k subtraction; it takes T as input rather than calling
# larft, so it shares no build edge with it and links only calaman.common
# (Side/Trans/Direct/StoreV) and calaman.error_handling, both one level up.
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
# calaman.common (Uplo) and calaman.error_handling, both added one level up.
add_subdirectory(pstf2)

# calaman.pstrf -- the blocked (level-3) pivoted Cholesky driver. Inlines the
# panel look-ahead over the wrapped BLAS (syrk trailing update) and falls back
# to calaman.pstf2 for small n or a degenerate block size; added after pstf2 so
# that target exists when it links it.
add_subdirectory(pstrf)

# calaman.sytrs -- solve A X = B for a symmetric matrix already Bunch-Kaufman
# factored (LAPACK's ?sytrs). A host walk of the pivot sequence over the wrapped
# BLAS (ger/geru/gemv/swap) plus its own D^-1 block applies (the one stage no
# BLAS expresses); complex is symmetric, not Hermitian. It shares no build edge
# with the pivoted-Cholesky pair above (a different factorization) and links
# only calaman.common and calaman.error_handling, both added one level up.
add_subdirectory(sytrs)

# calaman.sytrs2 -- the level-3 symmetric Bunch-Kaufman solve (LAPACK's
# ?sytrs2): the same solve as calaman.sytrs, but the interchanges batched to the
# ends so the triangular solves are trsm calls. A host composition over its own
# ?syconv VALUE move and D^-1 kernels; no build edge to calaman.sytrs (a
# parallel implementation, not a caller), and links only calaman.common and
# calaman.error_handling, both one level up.
add_subdirectory(sytrs2)

# calaman.sysv -- the symmetric indefinite solve driver (LAPACK's ?sysv): factor
# by the vendor Bunch-Kaufman wwr::sytrf, then solve with calaman.sytrs2 when
# workspace allows (else calaman.sytrs). It LINKS both solves (added above), so
# it follows them here.
add_subdirectory(sysv)

# calaman.hetrs -- the Hermitian cousin of calaman.sytrs (LAPACK ?hetrs): solve
# A X = B for a Hermitian matrix already Bunch-Kaufman factored. A host walk of
# the pivot sequence over wrapped BLAS (geru/gemv/swap) plus its own
# real-reciprocal 1x1, Hermitian 2x2 and row-conjugation device stages. Complex
# only. It shares no build edge with the symmetric solves above and links only
# calaman.common and calaman.error_handling, both one level up.
add_subdirectory(hetrs)

# calaman.hetf2 -- the unblocked Hermitian Bunch-Kaufman factorization (LAPACK
# ?hetf2), the from-scratch factorization the vendors omit for Hermitian (they
# ship only the symmetric ?sytrf). A host-driven pivot search over wrapped BLAS
# (iamax/her/scal) plus its own interchange and rank-2 device kernels. Complex
# only. It links only calaman.common and calaman.error_handling, both one level
# up.
add_subdirectory(hetf2)

# calaman.hetrs2 -- the level-3 Hermitian Bunch-Kaufman solve (LAPACK ?hetrs2):
# the same solve as calaman.hetrs, but the interchanges batched to the ends so
# the triangular solves are trsm calls (the second conjugate-transpose). A host
# composition over its own ?syconv VALUE move and Hermitian D^-1 kernels;
# complex only. It links only calaman.common and calaman.error_handling, both
# one level up.
add_subdirectory(hetrs2)

# calaman.hesv -- the Hermitian indefinite solve driver (LAPACK ?hesv): factor
# by calaman.hetf2, then solve with calaman.hetrs2 when workspace allows (else
# calaman.hetrs). It LINKS all three (added above), so it follows them here.
add_subdirectory(hesv)

# calaman.lasrt -- sort a 1-D array (increasing or decreasing), the GPU
# counterpart of LAPACK's ?lasrt. A single hand-launched device kernel
# (quicksort/insertion), structurally laset's twin but functionally standalone:
# it shares no build edge with the rest of src/ and links only calaman.common
# (SortDir) and calaman.error_handling, both added one level up.
add_subdirectory(lasrt)

# calaman.lanv2 -- Schur-standardise a real 2-by-2 block (LAPACK's ?lanv2), the
# 2-by-2 deflation step of the real Schur reduction. Pure host scalar arithmetic
# with no BLAS: structurally calaman.larfg's slimmest sibling (an exported
# header-only template) but taking only a stream, since it owns no device code.
# It links only wwr.runtime_api and calaman.error_handling (added one level up).
add_subdirectory(lanv2)

# calaman.ladiv -- real-arithmetic complex division (LAPACK's ?ladiv), the
# device scalar helper laln2 calls per-thread. The scalar routine is header-only
# and host/device; the module batches n of them on the device (calaman.lartg's
# sibling) so the oracle can exercise it on a card. Links only wwr.runtime_api
# and calaman.error_handling (added one level up).
add_subdirectory(ladiv)

# calaman.sym2x2 -- the header-only INTERFACE target carrying sym2x2.cuh, the
# device ports of ?lae2/?laev2/?lapy2 that sterf/steqr include per-thread.
# Added before calaman.lae2/laev2/lapy2 (their batched oracle wrappers, ladiv's
# siblings), which link it.
add_subdirectory(sym2x2)
add_subdirectory(lae2)
add_subdirectory(laev2)
add_subdirectory(lapy2)

# calaman.lapy3 / calaman.larmm -- header-only device ports of ?lapy3 and
# ?larmm (lapy3.cuh, larmm.cuh, each with its own ::header INTERFACE target),
# plus the batched oracle wrappers that exercise them on a card. lapy2's
# siblings; they link only wwr and calaman.error_handling.
add_subdirectory(lapy3)
add_subdirectory(larmm)

# calaman.sterf -- eigenvalues of a symmetric tridiagonal (LAPACK's ?sterf), one
# single-thread kernel over the sterf.cuh device core. That header links
# calaman::sym2x2, calaman::lanst::header and calaman::lasrt::header (all
# added above), so it follows them here.
add_subdirectory(sterf)

# calaman.steqr -- eigenpairs of a symmetric tridiagonal by implicit QL/QR
# (LAPACK's ?steqr). One single-block kernel that #includes the per-thread
# helpers of lanst, lartg, lasr and sym2x2, so it follows all four here.
add_subdirectory(steqr)

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
# calaman.error_handling (both added one level up).
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

# calaman.geev -- the ?geev driver: eigenvalues and optional left/right
# eigenvectors of a general real matrix. A host composition that chains the
# whole nonsymmetric-eigenproblem pipeline -- calaman.lange/calaman.lascl
# (optional scaling), calaman.gebal (balance), calaman.gehrd (Hessenberg), orghr
# (form Q) + calaman.hseqr (Schur form) or calaman.hseqr alone, calaman.trevc3
# (eigenvectors of the Schur form), calaman.gebak (undo balancing), plus a
# wwr::gemm back-transform and a wwr BLAS-1 normalization. Placed last: it links
# all of the above, so every one must already be defined.
add_subdirectory(geev)

# ?syev -- the real symmetric eigensolver driver; links lansy, lascl, sterf and
# steqr, so it follows them.
add_subdirectory(syev)

# calaman.tri_index -- the header-only INTERFACE target carrying tri_index.cuh,
# the device index maps of the triangular storage formats. Added before the
# conversion modules that link it.
add_subdirectory(tri_index)

# calaman.tpttr -- packed (tp) to full (tr) triangular storage; links tri_index.
add_subdirectory(tpttr)

# calaman.trttp -- full (tr) to packed (tp) triangular storage; links tri_index.
add_subdirectory(trttp)

# calaman.trttf -- full (tr) to RFP (tf) triangular storage; links tri_index.
add_subdirectory(trttf)

# calaman.tfttr -- RFP (tf) to full (tr) triangular storage; links tri_index.
add_subdirectory(tfttr)

# calaman.tpttf -- packed (tp) to RFP (tf) triangular storage; links tri_index.
add_subdirectory(tpttf)

# calaman.tfttp -- RFP (tf) to packed (tp) triangular storage; links tri_index.
add_subdirectory(tfttp)

# calaman.lansf -- the norm of a symmetric matrix in RFP storage (LAPACK's
# ?lansf); its device library links tri_index for the RFP map.
add_subdirectory(lansf)

# calaman.lanhf -- the norm of a Hermitian matrix in RFP storage (LAPACK's
# ?lanhf). No device code of its own: it runs calaman.lansf's kernel with the
# Hermitian flag and links calaman.lansf.device, so it follows lansf.
add_subdirectory(lanhf)

# calaman.lansp -- the norm of a symmetric matrix in packed storage (LAPACK's
# ?lansp); its device library links tri_index for the packed map.
add_subdirectory(lansp)

# calaman.lanhp -- the norm of a Hermitian matrix in packed storage (LAPACK's
# ?lanhp). No device code of its own: it runs calaman.lansp's kernel with the
# Hermitian flag and links calaman.lansp.device, so it follows lansp.
add_subdirectory(lanhp)

# calaman.lansb -- the norm of a symmetric band matrix in band storage (LAPACK's
# ?lansb). Links only calaman.common and calaman.error_handling.
add_subdirectory(lansb)

# calaman.lanhb -- the norm of a Hermitian band matrix in band storage (LAPACK's
# ?lanhb). No device code of its own: it runs calaman.lansb's kernel with the
# Hermitian flag and links calaman.lansb.device, so it follows lansb.
add_subdirectory(lanhb)

# calaman.lantr -- the norm of a triangular (trapezoidal) matrix (LAPACK's
# ?lantr); its device library, templated on the storage layout, links tri_index
# for the packed map and also serves lantp and lantb.
add_subdirectory(lantr)

# calaman.lantp / calaman.lantb -- the norm of a triangular matrix in packed /
# band storage (LAPACK's ?lantp / ?lantb). No device code of their own: they
# run calaman.lantr's kernel and link calaman.lantr.device, so they follow it.
add_subdirectory(lantp)
add_subdirectory(lantb)

# calaman.lagtm -- tridiagonal matrix-matrix multiply (LAPACK's ?lagtm). A leaf:
# links only calaman.common and calaman.error_handling.
add_subdirectory(lagtm)

# calaman.la_geamv -- y := alpha * |op(A)| * |x| + beta * |y| with the
# symbolic-zero nudge (LAPACK's ?la_geamv), the first extra-precise ?la_*amv
# kernel. A leaf: links only calaman.common and calaman.error_handling.
add_subdirectory(la_geamv)

# calaman.lacrm -- complex times real matrix multiply (LAPACK's ?lacrm): two
# real gemms around calaman.complex_cast's split_planes / merge_planes (added
# one level up), which calaman.larcm reuses.
add_subdirectory(lacrm)

# calaman.larcm -- real times complex matrix multiply (LAPACK's ?larcm): the
# mirror of calaman.lacrm, splitting the complex B instead of A.
add_subdirectory(larcm)

# calaman.rfp_blocks -- the host-side RFP sub-block geometry (offsets, leading
# dimensions, stored uplo of A11, A22 and the off-diagonal block), shared by
# the routines that run BLAS on RFP sub-blocks; calaman.hfrk (?hfrk and ?sfrk)
# is the first of them and follows it.
add_subdirectory(rfp_blocks)
add_subdirectory(hfrk)
# calaman.tfsm -- triangular solve with A in RFP (?tfsm), the second consumer.
add_subdirectory(tfsm)
add_subdirectory(lacn2)
