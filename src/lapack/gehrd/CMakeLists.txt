# calaman.gehrd -- LAPACK's ?gehrd, the blocked reduction of a general matrix to
# upper Hessenberg form by an orthogonal similarity Q^T*A*Q = H. A host
# composition: calaman.lahr2 reduces each panel (V/T/Y), one wrapped-BLAS gemm
# applies Y to the trailing block from the right, calaman.larfb applies the
# block reflector from the left, and calaman.gehd2 reduces the final tail. Pure
# host C++23 over those three modules plus gemm, so it owns no device code -- no
# .cu, no bridge header, the shape of calaman.gehd2 / calaman.lahr2.
#
# All module links are PUBLIC because gehrd() is an exported template DEFINED in
# the interface (like calaman.gehd2): a consumer instantiates the body, so every
# module the body names must reach it. calaman.error_handling is additionally
# re-exported (export import) because it supplies the Status return type, whose
# member functions a consumer must see. gehrd.cppm's GMF includes
# error_handling/error_macros.h (CLM_TRY), resolved root-relative via that
# target's exported include root.
#   wwr.wrappers.blas       gemm, trmm, axpy (the trailing + panel Y updates)
#   wwr.blas                wwrblasHandle_t, wwrblasGetStream, WWRBLAS_OP_*
#   wwr.runtime_api         wwrMemcpyAsync, wwrMemsetAsync, wwrStreamSynchronize
#   calaman.common          Side, Trans, Direct, StoreV (the larfb selectors)
#   calaman.gehd2           the unblocked tail
#   calaman.lahr2           the per-panel V/T/Y reduction
#   calaman.larfb           apply the block reflector from the left
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.gehrd
  PRIMARY_INTERFACE
  gehrd.cppm
  LINK_PUBLIC
  wwr.wrappers.blas
  wwr.blas
  wwr.runtime_api
  calaman.common
  calaman.gehd2
  calaman.lahr2
  calaman.larfb
  calaman.error_handling
  IMPORT_STD
)
