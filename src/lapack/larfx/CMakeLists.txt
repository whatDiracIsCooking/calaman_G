# calaman.larfx -- LAPACK's ?larfx, apply an elementary reflector
# H = I - tau*v*v^T to a matrix C in place, SPLIT on the reflector's order: a
# fused workspace-free kernel for order < 11, and a deferral to calaman.larf
# (one gemv + one ger) for order >= 11, exactly as reference ?larfx defers to
# ?larf. A shipped src/ module: written once against WarpWraps's `wwr*` names,
# it builds for either backend.

# The inline small-order device-kernel library -- hand-launched through
# wwr.extension.parallel_for from larfx.cu (one thread per column/row), so it
# links that INTERFACE library (which carries wwr.device + the bridge include
# root) PRIVATE, like calaman.laqp2's downdate. The link is PRIVATE, as every
# device-library link is.
calaman_add_gpu_device_library(
  NAME
  calaman.larfx.device
  SOURCES
  larfx.cu
  LINK_PRIVATE
  wwr::extension::parallel_for
)

# All module links are PUBLIC because larfx() is an exported template DEFINED in
# the interface (like calaman.larf): a consumer instantiates the body, so every
# module the body names -- calaman.larf (the fallback), the handle/stream types,
# and the device launcher reached through the GMF -- must reach it.
# calaman.error_handling is additionally re-exported (export import) because it
# supplies the Status return type, whose member functions a consumer must see.
#
# larfx.cppm's GMF includes larfx_bridge.h (INCLUDE_DIRS_PRIVATE, this dir),
# which pulls runtime.h -> selected_backend.h; a GMF cannot import, so
# wwrStream_t arrives by #include and the backend define is read via
# LINK_PRIVATE wwr_backend -- exactly the calaman.laqp2 arrangement.
#   wwr.blas                wwrblasHandle_t / wwrblasStatus_t / wwrblasGetStream
#   wwr.runtime_api         wwrStream_t, wwrGetLastError, wwrSuccess
#   calaman.common          Side (:enums, re-exported)
#   calaman.larf            calaman::larf -- the order >= 11 fallback
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.larfx
  PRIMARY_INTERFACE
  larfx.cppm
  LINK_PUBLIC
  wwr::blas
  wwr::runtime_api
  calaman::common
  calaman::larf
  calaman::error_handling
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::larfx<T>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the whole-archive link laqp2 needs.
calaman_link_device_archive(calaman.larfx calaman.larfx.device)
