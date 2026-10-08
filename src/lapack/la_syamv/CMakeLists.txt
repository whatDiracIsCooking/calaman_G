# calaman.la_syamv -- y := alpha * |A| * |x| + beta * |y| for a symmetric A
# held in one triangle, with the symbolic-zero nudge: LAPACK's ?la_syamv. Its
# kernel also serves calaman.la_heamv.

# The device-kernel library: one thread per element of y, over la_geamv's
# shared per-row body (calaman.la_amv), which carries calaman.common's src/
# root for common/align_up.h and common/enums.h too.
calaman_add_gpu_device_library(
  NAME
  calaman.la_syamv.device
  SOURCES
  la_syamv.cu
  LINK_PRIVATE
  calaman::la_amv
)

# The module: host C++23 over the launcher declared in la_syamv_bridge.h, as
# calaman.la_geamv's (INCLUDE_DIRS_PRIVATE for the GMF bridge, wwr::backend
# PRIVATE for runtime.h there, PUBLIC what the exported surface names).
calaman_add_cxx_module_library(
  NAME
  calaman.la_syamv
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

# Force the device archive members in, or the explicit device::la_syamv<T, R>
# instantiations are dropped and a consumer fails to link.
calaman_link_device_archive(calaman.la_syamv calaman.la_syamv.device)
