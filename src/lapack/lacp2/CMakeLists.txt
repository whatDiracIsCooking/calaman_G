# calaman.lacp2 -- copy all or part of a real matrix into a complex one on the
# device (imaginary part 0), LAPACK's ?lacp2: calaman.lacpy's shape with a
# widening store.

# The device-kernel library: one hand-launched per-element kernel. wwr.device
# carries runtime.h and complex.h; calaman.common the src/ root for
# "common/align_up.h", "common/elem_ops.cuh" (make_complex) and
# "common/enums.h".
calaman_add_gpu_device_library(
  NAME
  calaman.lacp2.device
  SOURCES
  lacp2.cu
  LINK_PRIVATE
  wwr::device
  calaman::common
)

# The module: host C++23 over the launcher declared in lacp2_bridge.h, which
# interface.cppm includes in its GMF (hence INCLUDE_DIRS_PRIVATE, and
# wwr_backend PRIVATE for the selected_backend.h behind runtime.h there).
# PUBLIC links are what the exported surface names: wwrStream_t, the complex
# types, Region / complex_fp (calaman.common) and Status.
calaman_add_cxx_module_library(
  NAME
  calaman.lacp2
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

# Force the device archive members in, or the explicit device::lacp2
# instantiations are dropped and a consumer fails to link.
calaman_link_device_archive(calaman.lacp2 calaman.lacp2.device)
