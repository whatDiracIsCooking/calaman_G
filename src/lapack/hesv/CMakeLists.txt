# calaman.hesv -- LAPACK's ?hesv, solve A X = B for a Hermitian indefinite
# matrix. The driver: factor by the Hermitian Bunch-Kaufman method
# (calaman.hetf2 -- ours, since the vendors ship no hetrf) then solve with
# calaman.hetrs. It owns no device code of its own -- the kernels arrive, with
# their whole-archive links, through the two modules it links. Complex only
# (c/z); real Hermitian is symmetric (?sysv).
#
# All links are PUBLIC because hesv()/hesv_bufferSize() are exported templates
# DEFINED in the interface (like sysv): a consumer instantiates the body, so
# every module the body names must reach it -- including calaman.hetf2 and
# calaman.hetrs, whose PUBLIC whole-archive device links then propagate here.
# calaman.error_handling is re-exported (export import) for the Status return
# type.
#   wwr.blas                handle, wwrblasGetStream / status
#   wwr.runtime_api         wwrStream_t, wwrMemcpy(Async), wwrStreamSynchronize
#   wwr.complex             the two complex element types (instantiation list)
#   calaman.common          Uplo, complex_fp
#   calaman.hetf2           the Hermitian Bunch-Kaufman factorization
#   calaman.hetrs           the level-2 solve against the factor
#   calaman.hetrs2          the level-3 solve (used when d_work >= n)
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.hesv
  PRIMARY_INTERFACE
  hesv.cppm
  LINK_PUBLIC
  wwr.blas
  wwr.runtime_api
  wwr.complex
  calaman.common
  calaman.hetf2
  calaman.hetrs
  calaman.hetrs2
  calaman.error_handling
  IMPORT_STD
)
