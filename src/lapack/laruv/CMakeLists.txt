# calaman.laruv -- LAPACK's uniform (0,1) generator ?laruv, bit for bit: one
# block of 128 threads, plus the block-cooperative laruv_block header
# (laruv.cuh) that ?larnv's kernels can call per chunk.

# The header's own target: the src/ root that resolves "lapack/laruv/laruv.cuh"
# (and the laruv_bridge.h it includes), plus wwr.device for runtime.h.
add_library(calaman.laruv.header INTERFACE)
add_library(calaman::laruv::header ALIAS calaman.laruv.header)
target_include_directories(
  calaman.laruv.header
  INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:${CALAMAN_INSTALL_INCLUDEDIR}>
)
target_link_libraries(calaman.laruv.header INTERFACE wwr::device)

# The device-kernel library: one hand-launched single-block kernel.
calaman_add_gpu_device_library(
  NAME
  calaman.laruv.device
  SOURCES
  laruv.cu
  LINK_PRIVATE
  calaman::laruv::header
)

# The module: host C++23 over the launcher declared in laruv_bridge.h, which
# interface.cppm includes in its GMF (hence INCLUDE_DIRS_PRIVATE, and
# wwr_backend PRIVATE for the selected_backend.h behind runtime.h there).
# PUBLIC links are what the exported surface names: wwrStream_t, real_fp
# (calaman.common) and Status (calaman.error_handling).
calaman_add_cxx_module_library(
  NAME
  calaman.laruv
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr::runtime_api
  calaman::common
  calaman::error_handling
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::laruv<T>
# instantiations are dropped and a consumer fails to link.
calaman_link_device_archive(calaman.laruv calaman.laruv.device)
