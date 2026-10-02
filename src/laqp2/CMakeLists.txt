# calaman.laqp2 -- LAPACK's ?laqp2, the unblocked level-2 Businger-Golub
# pivoted-QR panel. A host composition of calaman.larfg / calaman.larf plus the
# wrapped BLAS-1 iamax / swap / nrm2; its one device stage, the LAWN 176
# partial-norm downdate, is the device library below.

# The downdate device-kernel library -- hand-launched through
# wwr.extension.parallel_for from laqp2.cu, so it links that INTERFACE library
# (which carries wwr.device + the bridge include root) PRIVATE, like WarpWraps's
# own parallel_for consumers.
calaman_add_gpu_device_library(
  NAME
  calaman.laqp2.device
  SOURCES
  laqp2.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
)

# All module links are PUBLIC because laqp2() is an exported template DEFINED in
# the interface (like calaman.diff_norm): a consumer instantiates the body, so
# every module the body names (calaman.larfg, calaman.larf, the wrapped BLAS)
# must reach it. laqp2.cppm's GMF includes laqp2_bridge.h (INCLUDE_DIRS_PRIVATE,
# this directory), which pulls runtime.h -> selected_backend.h; a GMF
# cannot import, so wwrStream_t arrives by #include and the backend define is
# read via LINK_PRIVATE wwr_backend -- exactly the calaman.lacpy arrangement.
#   wwr.wrappers.blas   iamax, swap, nrm2
#   wwr.blas            wwrblasHandle_t / wwrblasStatus_t / WWRBLAS_STATUS_*
#   wwr.runtime_api     wwrMemcpy(Async), wwrStreamSynchronize, wwrSuccess
#   calaman.larfg       calaman::larfg
#   calaman.larf        calaman::larf, calaman::Side
calaman_add_cxx_module_library(
  NAME
  calaman.laqp2
  PRIMARY_INTERFACE
  laqp2.cppm
  LINK_PUBLIC
  wwr.wrappers.blas
  wwr.blas
  wwr.runtime_api
  calaman.larfg
  calaman.larf
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::laqp2_downdate<T>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the whole-archive link lacpy needs.
target_link_libraries(
  calaman.laqp2
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.laqp2.device>"
)
