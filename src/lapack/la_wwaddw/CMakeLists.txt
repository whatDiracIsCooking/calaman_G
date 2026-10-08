# calaman.la_wwaddw -- double-word accumulate, (x, y) <-- (x, y) + w, the GPU
# counterpart of LAPACK's ?la_wwaddw. Written once against WarpWraps's `wwr*`
# names. A leaf: no workspace, no other calaman routine.

# The device-kernel library, hand-launched like larscl2's; links PRIVATE as
# every device-library link is. wwr.device carries runtime.h; calaman.common
# carries the src/ root of "common/align_up.h" and "common/elem_ops.cuh".
calaman_add_gpu_device_library(
  NAME
  calaman.la_wwaddw.device
  SOURCES
  la_wwaddw.cu
  LINK_PRIVATE
  wwr::device
  calaman::common
)

# The module: host C++23. INCLUDE_DIRS_PRIVATE resolves la_wwaddw_bridge.h in
# the GMF, and wwr.backend lets the runtime.h it pulls in read the backend
# define. calaman.error_handling is re-exported for Status and carries
# error_macros.h.
calaman_add_cxx_module_library(
  NAME
  calaman.la_wwaddw
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr::runtime_api
  wwr::complex
  calaman::error_handling
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::la_wwaddw<T>
# instantiations are dropped and a consumer fails to link (as larscl2 does).
calaman_link_device_archive(calaman.la_wwaddw calaman.la_wwaddw.device)
