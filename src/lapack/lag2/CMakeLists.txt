# calaman.lag2 -- general matrix, double <-> single (LAPACK's dlag2s, slag2d,
# zlag2c, clag2z), plus the per-element lag2_convert header (lag2.cuh) that
# calaman.lat2's kernel shares.

# The header's own target: the src/ root that resolves "lapack/lag2/lag2.cuh",
# plus calaman.common for "common/elem_ops.cuh" (and, through it, wwr.device).
add_library(calaman.lag2.header INTERFACE)
add_library(calaman::lag2::header ALIAS calaman.lag2.header)
target_include_directories(
  calaman.lag2.header
  INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:${CALAMAN_INSTALL_INCLUDEDIR}>
)
target_link_libraries(calaman.lag2.header INTERFACE wwr::device calaman::common)

# The device-kernel library: one hand-launched per-element kernel.
calaman_add_gpu_device_library(
  NAME
  calaman.lag2.device
  SOURCES
  lag2.cu
  LINK_PRIVATE
  calaman::lag2::header
)

# The module: host C++23 over the launcher declared in lag2_bridge.h, which
# interface.cppm includes in its GMF (hence INCLUDE_DIRS_PRIVATE, and
# wwr_backend PRIVATE for the selected_backend.h behind runtime.h there).
# PUBLIC links are what the exported surface names: wwrStream_t, the complex
# types and Status (calaman.error_handling).
calaman_add_cxx_module_library(
  NAME
  calaman.lag2
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

# Force the device archive members in, or the explicit device::lag2
# instantiations are dropped and a consumer fails to link.
calaman_link_device_archive(calaman.lag2 calaman.lag2.device)
