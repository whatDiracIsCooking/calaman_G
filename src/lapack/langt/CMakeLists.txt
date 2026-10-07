# calaman.langt -- the norm of a general tridiagonal (dl, d, du), the GPU
# counterpart of LAPACK's ?langt.

# The device-kernel library: one single-block kernel, modeled on
# calaman.lanst's. calaman.common carries the src/ root for
# common/block_reduce.cuh, common/elem_ops.cuh and common/enums.h plus, through
# it, wwr.device (runtime.h, cooperative_groups.h, complex.h, the runtime libs).
calaman_add_gpu_device_library(
  NAME
  calaman.langt.device
  SOURCES
  langt.cu
  LINK_PRIVATE
  calaman::common
)

# The module: host C++23 over the launcher declared in langt_bridge.h, which
# interface.cppm includes in its GMF (hence INCLUDE_DIRS_PRIVATE, and
# wwr_backend PRIVATE for the selected_backend.h behind runtime.h there).
# PUBLIC links are what the exported surface names: wwrStream_t, the complex
# element types, MatrixNorm (re-exported from calaman.common) and Status
# (calaman.error_handling).
calaman_add_cxx_module_library(
  NAME
  calaman.langt
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

# Force the device archive members in, or the explicit device::langt<T, R>
# instantiations are dropped and a consumer fails to link -- as for lange.
calaman_link_device_archive(calaman.langt calaman.langt.device)
