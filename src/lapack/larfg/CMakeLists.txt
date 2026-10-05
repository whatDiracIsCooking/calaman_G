# calaman.larfg -- LAPACK's ?larfg, generate an elementary Householder
# reflector. Pure host C++23: ||x|| is a stock nrm2 and the tail scaling a stock
# scal, so this reaches the outermost WarpWraps layer (wwr.wrappers.blas) and
# owns no device code -- no .cu, no bridge header.
#
# All links are PUBLIC because larfg() is an exported template DEFINED in the
# interface (like calaman.diff_norm): a consumer instantiates the body itself,
# so every module the body names must reach it. IMPORT_STD for std::sqrt / abs.
# calaman.error_handling is additionally re-exported (export import) because it
# supplies the Status return type, whose member functions a consumer must see.
#   wwr.wrappers.blas   nrm2, scal
#   wwr.blas            wwrblasHandle_t / wwrblasStatus_t / WWRBLAS_STATUS_*
#   wwr.runtime_api     wwrMemcpy(Async) + wwrSuccess for the alpha read/write
#   calaman.error_handling  Status, the cross-domain return type
calaman_add_cxx_module_library(
  NAME
  calaman.larfg
  PRIMARY_INTERFACE
  larfg.cppm
  LINK_PUBLIC
  wwr::wrappers::blas
  wwr::blas
  wwr::runtime_api
  calaman::error_handling
  IMPORT_STD
)
