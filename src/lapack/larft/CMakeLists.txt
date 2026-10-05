# calaman.larft -- LAPACK's ?larft, form the triangular factor T of a block
# reflector H = I - V T V^T. A recursive host composition of wrapped BLAS-3
# (trmm, gemm) plus one geam per node for the off-diagonal block copy, so it
# reaches the outermost WarpWraps layer (wwr.wrappers.blas) and owns no device
# code.
#
# All links are PUBLIC because larft() is an exported template DEFINED in the
# interface (like calaman.larf): a consumer instantiates the body itself, so
# every module the body names must reach it. calaman.error_handling is
# additionally re-exported (export import) because it supplies the Status return
# type, whose member functions a consumer must see.
#   wwr.wrappers.blas       trmm, gemm, geam
#   wwr.blas                handle, status, WWRBLAS_OP_* / _SIDE_* / _FILL_*
#   wwr.runtime_api         wwrMemcpyAsync, wwrMemcpyDeviceToDevice
#   calaman.common          Direct, StoreV (:enums, re-exported)
#   calaman.error_handling  Status, the cross-domain return type
calaman_add_cxx_module_library(
  NAME
  calaman.larft
  PRIMARY_INTERFACE
  larft.cppm
  LINK_PUBLIC
  wwr::wrappers::blas
  wwr::blas
  wwr::runtime_api
  calaman::common
  calaman::error_handling
)
