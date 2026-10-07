# calaman.lapy3 -- batched device ?lapy3 (overflow-safe sqrt(x^2 + y^2 + z^2)),
# the oracle wrapper around the header-only lapy3_scalar in lapy3.cuh.
# calaman.lapy2's sibling.

# The header's own target: the src/ root, so a downstream .cu reaches
# "lapack/lapy3/lapy3.cuh" root-relative, plus wwr.device for its math.cuh.
add_library(calaman.lapy3.header INTERFACE)
add_library(calaman::lapy3::header ALIAS calaman.lapy3.header)
target_include_directories(
  calaman.lapy3.header
  INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:${CALAMAN_INSTALL_INCLUDEDIR}>
)
target_link_libraries(calaman.lapy3.header INTERFACE wwr::device)

calaman_add_gpu_device_library(
  NAME
  calaman.lapy3.device
  SOURCES
  lapy3.cu
  LINK_PRIVATE
  wwr::extension::parallel_for
  calaman::lapy3::header
)

# LINK_PUBLIC: what the exported surface names (wwrStream_t, Status).
# LINK_PRIVATE wwr_backend: the runtime.h lapy3_bridge.h pulls into the GMF
# reads the backend define.
calaman_add_cxx_module_library(
  NAME
  calaman.lapy3
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr::runtime_api
  calaman::error_handling
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Whole-archive, or the explicit device::lapy3<T> instantiations are dropped.
calaman_link_device_archive(calaman.lapy3 calaman.lapy3.device)
