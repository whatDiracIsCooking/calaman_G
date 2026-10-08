# calaman.larscl2 -- reciprocal diagonal row-scaling of a column-major matrix,
# x <-- D^-1 x, the GPU counterpart of LAPACK's ?larscl2 and the inverse twin
# of calaman.lascl2. Written once against WarpWraps's `wwr*` names.

# The device-kernel library, hand-launched like lascl2's; links PRIVATE as every
# device-library link is. wwr.device carries runtime.h; calaman.common carries
# the src/ root of "common/align_up.h" and "common/elem_ops.cuh".
calaman_add_gpu_device_library(
  NAME
  calaman.larscl2.device
  SOURCES
  larscl2.cu
  LINK_PRIVATE
  wwr::device
  calaman::common
)

# The module: host C++23. INCLUDE_DIRS_PRIVATE resolves larscl2_bridge.h in the
# GMF, and wwr.backend lets the runtime.h it pulls in read the backend define.
# calaman.common is PUBLIC for ComplexToRealType in the signature;
# calaman.error_handling is re-exported for Status and carries error_macros.h.
calaman_add_cxx_module_library(
  NAME
  calaman.larscl2
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

# Force the device archive members in, or the explicit device::larscl2<T>
# instantiations are dropped and a consumer fails to link (as lascl2 does).
calaman_link_device_archive(calaman.larscl2 calaman.larscl2.device)
