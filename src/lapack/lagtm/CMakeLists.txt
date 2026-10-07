# calaman.lagtm -- B := alpha * op(A) * X + beta * B for a general tridiagonal
# A, the GPU counterpart of LAPACK's ?lagtm.

# The device-kernel library: one per-element kernel. calaman.common carries the
# src/ root for common/align_up.h, common/elem_ops.cuh and common/enums.h plus,
# through it, wwr.device (runtime.h, complex.h, the runtime libs).
calaman_add_gpu_device_library(
  NAME
  calaman.lagtm.device
  SOURCES
  lagtm.cu
  LINK_PRIVATE
  calaman::common
)

# The module: host C++23 over the launcher declared in lagtm_bridge.h, which
# interface.cppm includes in its GMF (hence INCLUDE_DIRS_PRIVATE, and
# wwr_backend PRIVATE for the selected_backend.h behind runtime.h there).
# PUBLIC links are what the exported surface names: wwrStream_t, the complex
# element types, Trans (re-exported from calaman.common) and Status.
calaman_add_cxx_module_library(
  NAME
  calaman.lagtm
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

# Force the device archive members in, or the explicit device::lagtm<T, R>
# instantiations are dropped and a consumer fails to link -- as for langt.
calaman_link_device_archive(calaman.lagtm calaman.lagtm.device)
