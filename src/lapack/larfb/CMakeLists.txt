# calaman.larfb -- LAPACK's ?larfb, apply a block reflector H (or H^T) to a
# general matrix. A host composition of wrapped BLAS-3 (trmm, gemm) plus geam
# for the block copies and the rank-k subtraction, so it reaches the outermost
# WarpWraps layer (wwr.wrappers.blas) and owns no device code.
#
# All links are PUBLIC because larfb() is an exported template DEFINED in the
# interface (like calaman.larft): a consumer instantiates the body itself, so
# every module the body names must reach it. calaman.error_handling is
# additionally re-exported (export import) because it supplies the Status return
# type, whose member functions a consumer must see.
#   wwr.wrappers.blas       trmm, gemm, geam
#   wwr.blas                handle, status, WWRBLAS_OP_/_SIDE_/_FILL_/_DIAG_
#   calaman.common          Side, Trans, Direct, StoreV (:enums, re-exported)
#   calaman.error_handling  Status, the cross-domain return type
calaman_add_cxx_module_library(
  NAME
  calaman.larfb
  PRIMARY_INTERFACE
  larfb.cppm
  LINK_PUBLIC
  wwr::wrappers::blas
  wwr::blas
  calaman::common
  calaman::error_handling
)
