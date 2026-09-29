# calaman.lacpy -- copy all or part of a column-major matrix on the device, the
# GPU counterpart of LAPACK's ?lacpy. A shipped src/ module: it is written once
# against WarpWraps's `wwr*` names and builds for either backend.

# The device-kernel library. wwr.extension.parallel_for carries WarpWraps's src
# include root (so lacpy_bridge.h and the parallel_for.cuh it uses resolve) and
# wwr.device; the link is PRIVATE, as every device-library link is.
calaman_add_gpu_device_library(
  NAME
  calaman.lacpy.device
  SOURCES
  lacpy.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
)

# The module itself: ordinary host C++23, no device code. instantiations.cpp is
# the explicit-instantiation implementation unit paired with interface.cppm's
# `extern template` list.
#
# INCLUDE_DIRS_PRIVATE is for lacpy_bridge.h, which interface.cppm includes in
# its global module fragment. LINK_PRIVATE wwr_backend is for the
# gpu_stream_bridge.h that header pulls in there: a GMF cannot `import`, so
# wwrStream_t arrives by #include, and selected_backend.h behind it reads
# WWR_GPU_BACKEND_{CUDA,HIP} with no device pass to read __CUDACC__/__HIP__.
calaman_add_cxx_module_library(
  NAME
  calaman.lacpy
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr.runtime_api
  wwr.extension.runtime
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::launch_lacpy<T>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link. The install-interface spelling waits on
# the install tier (CALAMAN_INSTALL, docs/architecture.md §2); this build-
# interface link is what the tests need meanwhile.
target_link_libraries(
  calaman.lacpy
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.lacpy.device>"
)
