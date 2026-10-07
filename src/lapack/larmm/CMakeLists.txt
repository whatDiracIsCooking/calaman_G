# calaman.larmm -- batched device ?larmm (the scale that keeps C - A*B from
# overflowing, latrs3's safeguard), the oracle wrapper around the header-only
# larmm_scalar in larmm.cuh. calaman.lapy3's sibling.

# The header's own target: the src/ root, so a downstream .cu reaches
# "lapack/larmm/larmm.cuh" root-relative, plus wwr.device for the runtime.h that
# defines __forceinline__ under HIP. The body itself needs no math library.
add_library(calaman.larmm.header INTERFACE)
add_library(calaman::larmm::header ALIAS calaman.larmm.header)
target_include_directories(
  calaman.larmm.header
  INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:${CALAMAN_INSTALL_INCLUDEDIR}>
)
target_link_libraries(calaman.larmm.header INTERFACE wwr::device)

calaman_add_gpu_device_library(
  NAME
  calaman.larmm.device
  SOURCES
  larmm.cu
  LINK_PRIVATE
  wwr::extension::parallel_for
  calaman::larmm::header
)

# LINK_PUBLIC: what the exported surface names (wwrStream_t, Status).
# LINK_PRIVATE wwr_backend: the runtime.h larmm_bridge.h pulls into the GMF
# reads the backend define.
calaman_add_cxx_module_library(
  NAME
  calaman.larmm
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

# Whole-archive, or the explicit device::larmm<T> instantiations are dropped.
calaman_link_device_archive(calaman.larmm calaman.larmm.device)
