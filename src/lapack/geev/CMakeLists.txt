# calaman.geev -- LAPACK's ?geev, the driver that computes the eigenvalues and,
# optionally, the left/right eigenvectors of a general real matrix. A host
# composition in calaman.hseqr's shape: the O(1) control flow (the scaling
# decision, gebal's ilo/ihi, the per-vector normalization) runs on the host and
# stages the scalars it inspects through device memcpies, while every heavy step
# is a shipped kernel it dispatches to -- calaman.lange/calaman.lascl (the
# optional scaling of an out-of-range a), calaman.gebal (balancing),
# calaman.gehrd (Hessenberg reduction), calaman.orghr (form Q) + calaman.hseqr
# (Schur form) for the vector path else calaman.hseqr alone (eigenvalues),
# calaman.trevc3 (eigenvectors of the Schur form), calaman.gebak (undo
# balancing), with the back-transform V = Q*X one wwr::gemm per side and the
# final unit-2-norm + largest-real normalization wwr::nrm2 / scal / rot. Pure
# host C++23: no device code, no .cu, no bridge header. Placed last in src/ so
# every routine it dispatches to is already defined.
#
# All module links are PUBLIC because geev() is an exported template DEFINED in
# the interface (like calaman.hseqr): a consumer instantiates the body, so every
# module the body names must reach it. calaman.error_handling is additionally
# re-exported (export import) because it supplies the Status return type, whose
# member functions a consumer must see; geev.cppm's GMF includes
# error_handling/error_macros.h (CLM_TRY), resolved root-relative via that
# target's exported include root.
#   wwr.blas               wwrblasHandle_t / wwrblasGetStream / WWRBLAS_OP_N
#   wwr.solver             wwrsolverDnHandle_t (calaman.orghr's handle)
#   wwr.runtime_api        wwrMemcpyAsync / wwrStreamSynchronize (staging)
#   wwr.wrappers.blas      gemm (back-transform), nrm2 / scal / rot (normalize)
#   calaman.common         real_fp; Region / MatrixNorm / WorkspaceLayout
#   calaman.lange          max-abs norm for the scaling decision
#   calaman.lascl          scale a, rescale the eigenvalues
#   calaman.lacpy          copy the gehrd reflectors into Q
#   calaman.gebal          balancing (+ GebalJob)
#   calaman.gehrd          Hessenberg reduction
#   calaman.orghr          form the explicit Q (+ orghr_bufferSize)
#   calaman.hseqr          Schur form / eigenvalues (+ HseqrJob / HseqrCompz)
#   calaman.trevc3         eigenvectors of the Schur form
#   calaman.gebak          undo balancing (+ GebakJob / GebakSide)
#   calaman.error_handling Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.geev
  PRIMARY_INTERFACE
  geev.cppm
  LINK_PUBLIC
  wwr.blas
  wwr.solver
  wwr.runtime_api
  wwr.wrappers.blas
  calaman.common
  calaman.lange
  calaman.lascl
  calaman.lacpy
  calaman.gebal
  calaman.gehrd
  calaman.orghr
  calaman.hseqr
  calaman.trevc3
  calaman.gebak
  calaman.error_handling
  IMPORT_STD
)
