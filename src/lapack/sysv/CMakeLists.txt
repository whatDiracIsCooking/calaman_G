# calaman.sysv -- LAPACK's ?sysv, solve A X = B for a symmetric indefinite
# matrix. The driver: factor by the Bunch-Kaufman method (wwr::sytrf, the
# backend-neutral vendor routine) then solve with calaman.sytrs2 (level-3) or
# calaman.sytrs (level-2). It owns no device code of its own -- the
# factorization is one vendor call and the D^-1 kernels arrive, with their
# whole-archive links, through the two solve modules. Complex is SYMMETRIC (no
# conjugation); the
# Hermitian cousin is calaman.hesv.
#
# All links are PUBLIC because sysv()/sysv_bufferSize() are exported templates
# DEFINED in the interface (like pstrf): a consumer instantiates the body, so
# every module the body names must reach it -- including the two solve modules,
# whose PUBLIC whole-archive device links then propagate here.
# calaman.error_handling is re-exported (export import) for the Status type.
#   wwr.blas                handle, WWRBLAS_FILL_MODE_* / status
#   wwr.solver              wwrsolverDnHandle_t
#   wwr.runtime_api         wwrStream_t, wwrMemcpy(Async), wwrStreamSynchronize
#   wwr.complex             the two complex element types (instantiation list)
#   wwr.wrappers.solver     sytrf, sytrf_bufferSize
#   calaman.common          Uplo, usual_fp
#   calaman.sytrs           the level-2 solve against the Bunch-Kaufman factor
#   calaman.sytrs2          the level-3 solve (used when d_work >= n)
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
  calaman.sytrs2
  calaman.error_handling
  IMPORT_STD
)
