# calaman.pstrf -- LAPACK's ?pstrf, the blocked (level-3) pivoted Cholesky of a
# symmetric positive semidefinite matrix. A host composition of the wrapped BLAS
# syrk / swap / scal / gemv over the calaman.pstf2 unblocked fallback; its two
# non-BLAS stages -- the fused diagonal-and-pivot reduction and zeroing the
# trailing factor -- are the device library (structurally pstf2's, kept separate
# on the module boundary per issue #65).

# The device-kernel library. The pivot reduction is hand-launched; the
# trailing-factor zeroing rides wwr.extension.parallel_for, so it links that
# INTERFACE library (carrying wwr.device + the bridge include root) PRIVATE,
# like calaman.pstf2 / calaman.laqp2 / calaman.laqps.
calaman_add_gpu_device_library(
  NAME
  calaman.pstrf.device
  SOURCES
  pstrf.cu
  LINK_PRIVATE
  wwr::extension::parallel_for
)

# All module links are PUBLIC because pstrf() is an exported template DEFINED in
# the interface (like pstf2 / geqp3): a consumer instantiates the
# body, so every module the body names -- including calaman.pstf2, the unblocked
# fallback -- must reach it. pstrf.cppm's GMF includes pstrf_bridge.h
# (INCLUDE_DIRS_PRIVATE, this directory), which pulls runtime.h; a GMF cannot
# import, so wwrStream_t arrives by #include and the backend define is read via
# LINK_PRIVATE wwr_backend -- exactly the calaman.pstf2 arrangement.
#   wwr.wrappers.blas       swap, scal, gemv, syrk
#   wwr.blas                handle / status / WWRBLAS_OP_* / WWRBLAS_FILL_MODE_*
#   wwr.runtime_api         wwrMemcpy(Async), wwrMemsetAsync, wwrStreamSync
#   calaman.common          Uplo (the referenced-triangle selector)
#   calaman.pstf2           the unblocked level-2 fallback path
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.pstrf
  PRIMARY_INTERFACE
  pstrf.cppm
  LINK_PUBLIC
  wwr::wrappers::blas
  wwr::blas
  wwr::runtime_api
  calaman::common
  calaman::pstf2
  calaman::error_handling
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::pstrf_* kernel
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the whole-archive link lacpy needs.
calaman_link_device_archive(calaman.pstrf calaman.pstrf.device)
