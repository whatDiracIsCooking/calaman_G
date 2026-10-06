# calaman.lasr -- apply a sequence of plane rotations to a general matrix, the
# GPU counterpart of LAPACK's ?lasr, plus the block-cooperative lasr_block
# header (lasr.h) that ?steqr's kernel calls to update Z.

# The header's own target: an INTERFACE library carrying only the src/ root, so
# a downstream .cu reaches "lapack/lasr/lasr.h" (and the "common/enums.h" it
# includes) root-relative by linking calaman::lasr::header -- lanst.h's shape.
add_library(calaman.lasr.header INTERFACE)
add_library(calaman::lasr::header ALIAS calaman.lasr.header)
target_include_directories(
  calaman.lasr.header
  INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:${CALAMAN_INSTALL_INCLUDEDIR}>
)

# The device-kernel library: one launch of lasr_block per slab. calaman.common
# carries the src/ root for common/enums.h plus, through it, wwr.device.
calaman_add_gpu_device_library(
  NAME
  calaman.lasr.device
  SOURCES
  lasr.cu
  LINK_PRIVATE
  calaman::common
  calaman::lasr::header
)

# The module: host C++23 over the launcher declared in lasr_bridge.h, which
# interface.cppm includes in its GMF (hence INCLUDE_DIRS_PRIVATE, and
# wwr_backend PRIVATE for the selected_backend.h behind runtime.h there).
# PUBLIC links are what the exported surface names: wwrStream_t, Side/Pivot/
# Direct (re-exported from calaman.common) and Status (calaman.error_handling).
calaman_add_cxx_module_library(
  NAME
  calaman.lasr
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

# Force the device archive members in, or the explicit device::lasr<T>
# instantiations are dropped and a consumer fails to link -- as for lanst.
calaman_link_device_archive(calaman.lasr calaman.lasr.device)
