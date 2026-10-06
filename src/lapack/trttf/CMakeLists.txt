# calaman.trttf -- convert a triangular matrix from full (tr) to Rectangular
# Full Packed (tf) storage on the device, the GPU counterpart of LAPACK's ?trttf
# (s/d/c/z).

# The device-kernel library: one functor through wwr.extension.parallel_for
# (which carries wwr.device), indexing through calaman::tri_index's RFP map.
# PRIVATE, as every device-library link is; calaman.common resolves
# "common/enums.h" and "common/elem_ops.cuh" (the conjugate).
calaman_add_gpu_device_library(
  NAME
  calaman.trttf.device
  SOURCES
  trttf.cu
  LINK_PRIVATE
  wwr::extension::parallel_for
  calaman::tri_index
  calaman::common
)

# The module: host C++23 only. LINK_PUBLIC is every module the exported surface
# names; wwr_backend is PRIVATE for the runtime.h trttf_bridge.h pulls into the
# GMF; INCLUDE_DIRS_PRIVATE resolves the bridge itself.
calaman_add_cxx_module_library(
  NAME
  calaman.trttf
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

# Whole-archive the device members, or the explicit device::trttf<T>
# instantiations are dropped and a consumer fails to link.
calaman_link_device_archive(calaman.trttf calaman.trttf.device)
