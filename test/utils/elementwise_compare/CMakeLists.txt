# calaman.test.elementwise_compare -- a GPU test utility: count differing
# elements of two device arrays. Built as a library the suites under test/test/
# link; its own self-test lives at test/test/elementwise_compare/. Not shipped:
# it lives under test/ and nothing installs it, so the WHOLE_ARCHIVE wiring
# below carries no $<INSTALL_INTERFACE:> spelling, unlike a src/ module.

# The device-kernel library. wwr.extension.parallel_for carries WarpWraps's src
# include root (so elementwise_compare_bridge.h and the parallel_for.cuh it uses
# resolve) and wwr.device; the link is PRIVATE, as every device-library link is.
calaman_add_gpu_device_library(
  NAME
  calaman.test.elementwise_compare.device
  SOURCES
  elementwise_compare.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
)

# The module itself: ordinary host C++23, no device code. instantiations.cpp is
# the explicit-instantiation implementation unit paired with interface.cppm's
# `extern template` list.
#
# INCLUDE_DIRS_PRIVATE is for elementwise_compare_bridge.h, which interface.cppm
# includes in its global module fragment. LINK_PRIVATE wwr_backend is for the
# gpu_stream_bridge.h that header pulls in there: a GMF cannot `import`, so
# wwrStream_t arrives by #include, and selected_backend.h behind it reads
# WWR_GPU_BACKEND_{CUDA,HIP} with no device pass to read __CUDACC__/__HIP__.
calaman_add_cxx_module_library(
  NAME
  calaman.test.elementwise_compare
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr.runtime_api
  wwr.extension.memory_buffer
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit
# device::write_mismatch_flags<T> instantiations and reduce_count are dropped
# (the module only names them through lazily-resolved calls) and a consumer
# fails to link. Build-interface only -- this target is never installed.
target_link_libraries(
  calaman.test.elementwise_compare
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.test.elementwise_compare.device>"
)
