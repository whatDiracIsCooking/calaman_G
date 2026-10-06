# calaman.aberth -- every root of a batch of polynomials by the Aberth-Ehrlich
# iteration, one warp per polynomial. Not a LAPACK routine, so its own module.

# The header's own target: aberth.cuh (aberth_warp, the warp-level core) for a
# downstream .cu that solves a polynomial inside its own kernel. The src/ root
# spells it "aberth/aberth.cuh"; calaman.common carries the common/ headers it
# includes and, through them, wwr.device.
add_library(calaman.aberth.header INTERFACE)
add_library(calaman::aberth::header ALIAS calaman.aberth.header)
target_include_directories(
  calaman.aberth.header
  INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:${CALAMAN_INSTALL_INCLUDEDIR}>
)
target_link_libraries(calaman.aberth.header INTERFACE calaman::common)

# The device-kernel library: the batched launch over aberth_warp.
calaman_add_gpu_device_library(
  NAME
  calaman.aberth.device
  SOURCES
  aberth.cu
  LINK_PRIVATE
  calaman::aberth::header
)

# The module: host C++23 only. LINK_PUBLIC is what the exported surface names;
# LINK_PRIVATE wwr_backend serves the runtime.h and fp_types.h that
# aberth_bridge.h pulls into the global module fragment.
calaman_add_cxx_module_library(
  NAME
  calaman.aberth
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

# Whole-archive, or the explicit device::aberth instantiations are dropped and a
# consumer fails to link (as for every module here with a device library).
calaman_link_device_archive(calaman.aberth calaman.aberth.device)
