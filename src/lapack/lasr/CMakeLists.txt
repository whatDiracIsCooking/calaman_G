# calaman.lasr -- apply a sequence of plane rotations to a general matrix, the
# GPU counterpart of LAPACK's ?lasr, plus the block-cooperative lasr_block
# header (lasr.h) that ?steqr's kernel calls to update Z.

# The header's own target: the src/ root that resolves "lapack/lasr/lasr.h"
# (and the "common/enums.h" it includes), plus calaman.common for
# "common/elem_ops.cuh" and, through it, wwr.device -- lag2.cuh's shape.
add_library(calaman.lasr.header INTERFACE)
add_library(calaman::lasr::header ALIAS calaman.lasr.header)
target_include_directories(
  calaman.lasr.header
  INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:${CALAMAN_INSTALL_INCLUDEDIR}>
)
target_link_libraries(calaman.lasr.header INTERFACE wwr::device calaman::common)

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
# Direct and ComplexToRealType (calaman.common), wwrFloatComplex/
# wwrDoubleComplex (wwr.complex) and Status (calaman.error_handling).
calaman_add_cxx_module_library(
  NAME
  calaman.lasr
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

# Force the device archive members in, or the explicit device::lasr<T>
# instantiations are dropped and a consumer fails to link -- as for lanst.
calaman_link_device_archive(calaman.lasr calaman.lasr.device)
