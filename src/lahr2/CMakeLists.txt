# calaman.lahr2 -- LAPACK's ?lahr2, reduce the first nb columns of a panel to
# upper Hessenberg form and return the V/T/Y that ?gehrd applies to the trailing
# matrix. A host composition of wrapped BLAS (gemv/trmv/gemm/trmm/geam/copy/
# axpy/scal) plus calaman.larfg per column, so it reaches the outermost
# WarpWraps layer (wwr.wrappers.blas) and owns no device code -- no .cu, no
# bridge header, the shape of calaman.larft.
#
# All links are PUBLIC because lahr2() is an exported template DEFINED in the
# interface (like calaman.larfg): a consumer instantiates the body itself, so
# every module the body names must reach it. calaman.error_handling is
# additionally re-exported (export import) because it supplies the Status return
# type, whose member functions a consumer must see.
#   wwr.wrappers.blas       gemv, trmv, gemm, trmm, geam, copy, axpy, scal
#   wwr.blas                handle, status, WWRBLAS_OP_*/_SIDE_*/_FILL_*/_DIAG_*
#   wwr.runtime_api         wwrMemcpyAsync, wwrMemcpyHostToDevice, wwrStreamSync
#   calaman.larfg           calaman::larfg, the per-column reflector
#   calaman.error_handling  Status, the cross-domain return type
calaman_add_cxx_module_library(
  NAME
  calaman.lahr2
  PRIMARY_INTERFACE
  lahr2.cppm
  LINK_PUBLIC
  wwr.wrappers.blas
  wwr.blas
  wwr.runtime_api
  calaman.larfg
  calaman.error_handling
)
