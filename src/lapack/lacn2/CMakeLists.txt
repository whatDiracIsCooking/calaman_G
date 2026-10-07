# calaman.lacn2 -- LAPACK's ?lacn2, the Hager/Higham 1-norm estimator driven by
# reverse communication (s/d). A shipped src/ module: written once against
# WarpWraps's `wwr*` names, built for either backend.

# The device-kernel library: five elementwise passes launched through
# wwr.extension.parallel_for, linked PRIVATE for parallel_for.cuh and,
# transitively, the bridge's runtime.h.
calaman_add_gpu_device_library(
  NAME
  calaman.lacn2.device
  SOURCES
  lacn2.cu
  LINK_PRIVATE
  wwr::extension::parallel_for
)

# The host state machine. interface.cppm's GMF includes lacn2_bridge.h
# (INCLUDE_DIRS_PRIVATE, this dir), which pulls runtime.h; a GMF cannot import,
# so the backend define is read via LINK_PRIVATE wwr::backend -- the rscl
# arrangement. PUBLIC links are the modules the interface imports:
#   wwr.blas                handle, status codes, wwrblasGetStream
#   wwr.wrappers.blas       asum, iamax, copy
#   wwr.runtime_api         wwrMemcpyAsync, wwrMemsetAsync, wwrStreamSynchronize
#   calaman.common          real_fp, WorkspaceLayout, carve_workspace
#   calaman.error_handling  Status (re-exported) and CLM_TRY's include root
calaman_add_cxx_module_library(
  NAME
  calaman.lacn2
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr::blas
  wwr::wrappers::blas
  wwr::runtime_api
  calaman::common
  calaman::error_handling
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::lacn2_*
# instantiations are dropped and a consumer fails to link.
calaman_link_device_archive(calaman.lacn2 calaman.lacn2.device)
