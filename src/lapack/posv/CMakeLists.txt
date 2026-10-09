# calaman.posv -- LAPACK's ?posv, solve A X = B for a Hermitian/symmetric
# positive definite matrix. The driver: Cholesky-factor with wwr::potrf, then
# solve with wwr::potrs -- both backend-neutral vendor routines, so the module
# owns no device code of its own.
#
# All links are PUBLIC because posv()/posv_bufferSize() are exported templates
# DEFINED in the interface: a consumer instantiates the body, so every module
# the body names must reach it. calaman.error_handling is re-exported (export
# import) for the Status type.
#   wwr.blas                WWRBLAS_FILL_MODE_* / status
#   wwr.solver              wwrsolverDnHandle_t
#   wwr.runtime_api         wwrStream_t, wwrMemcpyAsync, wwrStreamSynchronize
#   wwr.complex             the two complex element types (instantiation list)
#   wwr.wrappers.solver     potrf, potrf_bufferSize, potrs
#   calaman.common          Uplo, usual_fp
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.posv
  PRIMARY_INTERFACE
  posv.cppm
  LINK_PUBLIC
  wwr::blas
  wwr::solver
  wwr::runtime_api
  wwr::complex
  wwr::wrappers::solver
  calaman::common
  calaman::error_handling
  IMPORT_STD
)
