# calaman.columnwise_ell1 -- the L1 norm of each column of a column-major
# matrix, in one kernel. A shipped src/ module, structurally the twin of
# calaman.lacpy: a host wrapper over a device kernel, written once against
# WarpWraps's `wwr*` names and built for either backend.
#
# Per-column L1 norm is not a LAPACK routine (it is the per-column form of the
# asum calaman.diff_norm wraps), so it is its own module rather than a
# calaman.linalg partition.

# The device-kernel library. The whole computation is a per-column reduction, so
# columnwise_ell1.cu owns no kernel itself -- it #includes
# calaman.reduce_columns's header (reduce_columns/reduce_columns.cuh) and
# launches that. Linking calaman.reduce_columns PRIVATE carries the src/ include
# root for that header plus its own device deps (wwr.device, wwr.extension.bridge
# -- runtime.cuh, WWR_WARP_SIZE, and the gpu_stream_bridge.h that
# columnwise_ell1_bridge.h also pulls in). The link is PRIVATE, as every
# device-library link is.
calaman_add_gpu_device_library(
  NAME
  calaman.columnwise_ell1.device
  SOURCES
  columnwise_ell1.cu
  LINK_PRIVATE
  calaman.reduce_columns
)

# The module itself: ordinary host C++23, no device code. instantiations.cpp is
# the explicit-instantiation implementation unit paired with interface.cppm's
# `extern template` list.
#
# INCLUDE_DIRS_PRIVATE is for columnwise_ell1_bridge.h, which interface.cppm
# includes in its global module fragment. LINK_PRIVATE wwr_backend is for the
# gpu_stream_bridge.h that header pulls in there: a GMF cannot `import`, so
# wwrStream_t arrives by #include, and selected_backend.h behind it reads
# WWR_GPU_BACKEND_{CUDA,HIP} with no device pass to read __CUDACC__/__HIP__.
# LINK_PUBLIC wwr.runtime_api is the wwrStream_t the exported signature names.
calaman_add_cxx_module_library(
  NAME
  calaman.columnwise_ell1
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr.runtime_api
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::columnwise_ell1<T>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the same whole-archive link lacpy and
# horner need. The install-interface spelling waits on the install tier
# (CALAMAN_INSTALL, docs/architecture.md §2); this build-interface link is what
# the tests need meanwhile.
target_link_libraries(
  calaman.columnwise_ell1
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.columnwise_ell1.device>"
)
