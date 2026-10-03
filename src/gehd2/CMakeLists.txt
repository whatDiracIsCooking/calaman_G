# calaman.gehd2 -- LAPACK's ?gehd2, the unblocked reduction of a general matrix
# to upper Hessenberg form by an orthogonal similarity Q^T*A*Q = H. A host
# composition of calaman.larfg (one reflector per column) and calaman.larf1f
# (its right then left application); the slim sibling of calaman.laqp2. Pure
# host C++23 over those two modules, so it owns no device code.
#
# All module links are PUBLIC because gehd2() is an exported template DEFINED in
# the interface (like calaman.laqp2): a consumer instantiates the body, so every
# module the body names must reach it. calaman.error_handling is additionally
# re-exported (export import) because it supplies the Status return type, whose
# member functions a consumer must see. gehd2.cppm's GMF includes
# error_handling/error_macros.h (CLM_TRY), resolved root-relative via that
# target's exported include root.
#   wwr.blas                wwrblasHandle_t, wwrblasGetStream, WWRBLAS_STATUS_*
#   wwr.runtime_api         wwrMemcpyAsync, wwrStreamSynchronize, bad-arg status
#   calaman.larfg           calaman::larfg
#   calaman.larf1f          calaman::larf1f, calaman::Side
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.gehd2
  PRIMARY_INTERFACE
  gehd2.cppm
  LINK_PUBLIC
  wwr.blas
  wwr.runtime_api
  calaman.larfg
  calaman.larf1f
  calaman.error_handling
  IMPORT_STD
)
