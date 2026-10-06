# calaman.lat2 -- triangular matrix, double to single (LAPACK's dlat2s and
# zlat2c): calaman.lag2's per-element conversion (lag2.cuh) restricted to one
# triangle.

# The device-kernel library: one hand-launched per-element kernel.
# lag2::header carries "lapack/lag2/lag2.cuh", calaman.common and wwr.device.
calaman_add_gpu_device_library(
  NAME
  calaman.lat2.device
  SOURCES
  lat2.cu
  LINK_PRIVATE
  calaman::lag2::header
)

# The module: host C++23 over the launcher declared in lat2_bridge.h, which
# interface.cppm includes in its GMF (hence INCLUDE_DIRS_PRIVATE, and
# wwr_backend PRIVATE for the selected_backend.h behind runtime.h there).
# PUBLIC links are what the exported surface names: wwrStream_t, the complex
# types, Uplo (calaman.common) and Status (calaman.error_handling).
calaman_add_cxx_module_library(
  NAME
  calaman.lat2
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

# Force the device archive members in, or the explicit device::lat2
# instantiations are dropped and a consumer fails to link.
calaman_link_device_archive(calaman.lat2 calaman.lat2.device)
