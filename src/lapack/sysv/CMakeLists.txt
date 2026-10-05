# calaman.sysv -- LAPACK's ?sysv, solve A X = B for a symmetric indefinite
# matrix. The driver: factor by the Bunch-Kaufman method (wwr::sytrf, the
# backend-neutral vendor routine) then solve with calaman.sytrs. It owns no
# device code of its own -- the factorization is one vendor call and the D^-1
# kernels arrive, with their whole-archive link, through calaman.sytrs. Complex
# is SYMMETRIC (no conjugation); the Hermitian cousin is calaman.hesv.
#
# All links are PUBLIC because sysv()/sysv_bufferSize() are exported templates
# DEFINED in the interface (like pstrf): a consumer instantiates the body, so
# every module the body names must reach it -- including calaman.sytrs, whose
# PUBLIC whole-archive device link then propagates here. calaman.error_handling
# is re-exported (export import) for the Status return type.
#   wwr.blas                handle, WWRBLAS_FILL_MODE_* / status
#   wwr.solver              wwrsolverDnHandle_t
#   wwr.runtime_api         wwrStream_t, wwrMemcpy(Async), wwrStreamSynchronize
#   wwr.complex             the two complex element types (instantiation list)
#   wwr.wrappers.solver     sytrf, sytrf_bufferSize
#   calaman.common          Uplo, usual_fp
#   calaman.sytrs           the solve against the Bunch-Kaufman factor
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.sysv
  PRIMARY_INTERFACE
  sysv.cppm
  LINK_PUBLIC
  wwr.blas
  wwr.solver
  wwr.runtime_api
  wwr.complex
  wwr.wrappers.solver
  calaman.common
  calaman.sytrs
  calaman.error_handling
  IMPORT_STD
)
