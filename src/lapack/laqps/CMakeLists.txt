# calaman.laqps -- LAPACK's ?laqps, the blocked level-3 Businger-Golub
# pivoted-QR panel. A host composition of calaman.larfg plus the wrapped BLAS
# (iamax, swap, gemv, gemm, nrm2) that accumulates the auxiliary matrix F and
# defers the trailing-matrix update to one gemm per block; its one device stage,
# the deferred LAWN 176 partial-norm downdate (plus a one-shot flag clear), is
# the device library below.

# The downdate device-kernel library -- hand-launched through
# wwr.extension.parallel_for from laqps.cu, so it links that INTERFACE library
# (which carries wwr.device + the bridge include root) PRIVATE, like WarpWraps's
# own parallel_for consumers.
calaman_add_gpu_device_library(
  NAME
  calaman.laqps.device
  SOURCES
  laqps.cu
  LINK_PRIVATE
  wwr::extension::parallel_for
)

# All module links are PUBLIC because laqps() is an exported template DEFINED in
# the interface (like calaman.diff_norm): a consumer instantiates the body, so
# every module the body names (calaman.larfg, the wrapped BLAS) must reach it.
# laqps.cppm's GMF includes laqps_bridge.h (INCLUDE_DIRS_PRIVATE, this dir),
# which pulls runtime.h -> selected_backend.h; a GMF cannot import, so
# wwrStream_t arrives by #include and the backend define is read via
# LINK_PRIVATE wwr_backend -- the calaman.lacpy / calaman.laqp2 arrangement.
#   wwr.wrappers.blas       iamax, swap, nrm2, gemv, gemm
#   wwr.blas                handle, status, WWRBLAS_OP_* / WWRBLAS_STATUS_*
#   wwr.runtime_api         wwrMemcpy(Async), wwrStreamSynchronize, wwrSuccess
#   calaman.larfg           calaman::larfg
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.laqps
  PRIMARY_INTERFACE
  laqps.cppm
  LINK_PUBLIC
  wwr::wrappers::blas
  wwr::blas
  wwr::runtime_api
  calaman::larfg
  calaman::error_handling
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::laqps_downdate<T>
# / device::laqps_clear_flags instantiations are dropped (the module only names
# them through lazily-resolved calls) and a consumer fails to link -- the same
# whole-archive link lacpy needs.
calaman_link_device_archive(calaman.laqps calaman.laqps.device)
