# calaman.lassq -- the scaled sum of squares of a strided vector, the GPU
# counterpart of LAPACK's ?lassq, over all four element types. A shipped src/
# module: written once against WarpWraps's `wwr*` names, built for either
# backend.

# The device-kernel library: one single-block kernel. calaman.common carries the
# src/ root for common/block_reduce.cuh and common/elem_ops.cuh plus, through
# it, wwr.device (runtime.h, cooperative_groups.h, complex.h, math.cuh and the
# runtime libs). The link is PRIVATE, as every device-library link is.
calaman_add_gpu_device_library(
  NAME
  calaman.lassq.device
  SOURCES
  lassq.cu
  LINK_PRIVATE
  calaman::common
)

# The module: host C++23 over the launcher declared in lassq_bridge.h, which
# interface.cppm includes in its GMF (hence INCLUDE_DIRS_PRIVATE, and
# wwr_backend PRIVATE for the selected_backend.h behind runtime.h there).
# PUBLIC links are what the exported surface names: wwrStream_t, the complex
# element types, ComplexToRealType (calaman.common) and Status
# (calaman.error_handling, re-exported through `export import`).
calaman_add_cxx_module_library(
  NAME
  calaman.lassq
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

# Force the device archive members in, or the explicit device::lassq<T, R>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- as for lanst.
calaman_link_device_archive(calaman.lassq calaman.lassq.device)
