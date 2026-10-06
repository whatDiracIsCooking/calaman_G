# calaman.lae2 -- batched device ?lae2 (eigenvalues of a symmetric 2x2), the
# oracle wrapper around lae2_scalar from calaman.sym2x2's shared device header.
# Structurally calaman.ladiv's sibling: one per-element functor through
# wwr.extension.parallel_for.

# PRIVATE calaman::sym2x2 supplies "lapack/sym2x2/sym2x2.cuh" and its math.cuh.
calaman_add_gpu_device_library(
  NAME
  calaman.lae2.device
  SOURCES
  lae2.cu
  LINK_PRIVATE
  wwr::extension::parallel_for
  calaman::sym2x2
)

# LINK_PUBLIC: what the exported surface names (wwrStream_t, Status).
# LINK_PRIVATE wwr_backend: the runtime.h lae2_bridge.h pulls into the GMF
# reads the backend define.
calaman_add_cxx_module_library(
  NAME
  calaman.lae2
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

# Whole-archive, or the explicit device::lae2<T> instantiations are dropped.
calaman_link_device_archive(calaman.lae2 calaman.lae2.device)
