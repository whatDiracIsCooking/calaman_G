# calaman.larf1f -- LAPACK's ?larf1f, apply an elementary reflector
# H = I - tau*v*v^T to a matrix C in place with v(1) an implicit 1. It is
# calaman.larf's algebra with the unit head special-cased: one gemv + one ger
# over the stored tail, two axpy folding in v(1)=1, and a scal fast-path when
# the reflector is the bare unit. Pure host C++23 over stock BLAS-1/2, so it
# reaches the outermost WarpWraps layer (wwr.wrappers.blas) and owns no device
# code.
#
# All links are PUBLIC because larf1f() is an exported template DEFINED in the
# interface (like calaman.larf): a consumer instantiates the body itself, so
# every module the body names must reach it. calaman.error_handling is
# additionally re-exported (export import) because it supplies the Status return
# type, whose member functions a consumer must see.
#   wwr.wrappers.blas       gemv, ger, axpy, scal
#   wwr.blas                handle, status, WWRBLAS_OP_* / WWRBLAS_STATUS_*
#   calaman.common          kZero<T>, kOne<T>; Side (:enums, re-exported)
#   calaman.error_handling  Status, the cross-domain return type
calaman_add_cxx_module_library(
  NAME
  calaman.larf1f
  PRIMARY_INTERFACE
  larf1f.cppm
  LINK_PUBLIC
  wwr.wrappers.blas
  wwr.blas
  calaman.common
  calaman.error_handling
)
