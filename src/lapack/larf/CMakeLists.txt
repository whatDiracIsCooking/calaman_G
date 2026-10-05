# calaman.larf -- LAPACK's ?larf, apply an elementary reflector
# H = I - tau*v*v^T to a matrix C in place, as one gemv plus one ger. Pure host
# C++23 over stock BLAS-2, so it reaches the outermost WarpWraps layer
# (wwr.wrappers.blas) and owns no device code.
#
# All links are PUBLIC because larf() is an exported template DEFINED in the
# interface (like calaman.diff_norm): a consumer instantiates the body itself,
# so every module the body names must reach it. calaman.error_handling is
# additionally re-exported (export import) because it supplies the Status return
# type, whose member functions a consumer must see.
#   wwr.wrappers.blas       gemv, ger
#   wwr.blas                handle, status, WWRBLAS_OP_* / WWRBLAS_STATUS_*
#   calaman.common          kZero<T>, kOne<T>; Side (:enums, re-exported)
#   calaman.error_handling  Status, the cross-domain return type
calaman_add_cxx_module_library(
  NAME
  calaman.larf
  PRIMARY_INTERFACE
  larf.cppm
  LINK_PUBLIC
  wwr::wrappers::blas
  wwr::blas
  calaman::common
  calaman::error_handling
)
