# calaman.laev2 -- batched device ?laev2 (eigenvalues and eigenvector rotation
# of a symmetric 2x2), the oracle wrapper around laev2_scalar from
# calaman.sym2x2's shared device header. calaman.lae2's twin.

# PRIVATE calaman::sym2x2 supplies "lapack/sym2x2/sym2x2.cuh" and its math.cuh.
calaman_add_gpu_device_library(
  NAME
  calaman.laev2.device
  SOURCES
  laev2.cu
  LINK_PRIVATE
  wwr::extension::parallel_for
  calaman::sym2x2
)

# LINK_PUBLIC: what the exported surface names (wwrStream_t, Status).
# LINK_PRIVATE wwr_backend: the runtime.h laev2_bridge.h pulls into the GMF
# reads the backend define.
calaman_add_cxx_module_library(
  NAME
  calaman.laev2
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

# Whole-archive, or the explicit device::laev2<T> instantiations are dropped.
calaman_link_device_archive(calaman.laev2 calaman.laev2.device)
