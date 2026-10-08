# calaman.ilalc -- the last non-zero column of a general matrix, the GPU
# counterpart of LAPACK's ila?lc. Written once against WarpWraps's `wwr*`
# names and built for either backend.

# The device-kernel library, links PRIVATE as every device-library link is.
# calaman.reduce_columns is the stage-two max fold (and carries wwr.device for
# runtime.h); calaman.common carries the src/ root of "common/elem_ops.cuh".
calaman_add_gpu_device_library(
  NAME
  calaman.ilalc.device
  SOURCES
  ilalc.cu
  LINK_PRIVATE
  calaman::reduce_columns
  calaman::common
)

# The module: host C++23. INCLUDE_DIRS_PRIVATE resolves ilalc_bridge.h in the
# GMF, and wwr.backend lets the runtime.h it pulls in read the backend define.
# calaman.common is PUBLIC for WorkspaceLayout in the interface;
# calaman.error_handling is re-exported for Status and carries error_macros.h.
calaman_add_cxx_module_library(
  NAME
  calaman.ilalc
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr::runtime_api
  wwr::complex
  calaman::common
  calaman::error_handling
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::ilalc<T>
# instantiations are dropped and a consumer fails to link (as lange does).
calaman_link_device_archive(calaman.ilalc calaman.ilalc.device)
