# calaman.columnwise_ell2 -- the L2 (Euclidean) norm of each column of a
# column-major matrix. A shipped src/ module, the twin of
# calaman.columnwise_ell1 (and structurally of calaman.lacpy): a host wrapper
# over a device kernel, written once against WarpWraps's `wwr*` names and built
# for either backend.
#
# Per-column L2 norm is not a LAPACK routine (it is the per-column form of the
# nrm2 calaman.diff_norm wraps), so it is its own module rather than a
# calaman.linalg partition.

# The device-kernel library. columnwise_ell2.cu owns no reduction kernel itself:
# pass one #includes calaman.reduce_columns's header
# (reduce_columns/reduce_columns.cuh) and launches that for the per-column sum
# of squares; pass two #includes wwr.extension.parallel_for's header
# (extension/parallel_for/parallel_for.cuh) and launches it for the elementwise
# sqrt of the result. So it links BOTH, PRIVATE as every device-library link is:
#   calaman.reduce_columns        the segmented reduce (pass one)
#   wwr.extension.parallel_for    the index-per-thread map (pass two)
# calaman.reduce_columns carries the src/ include root for its header plus its
# own device deps -- wwr.device, which provides runtime.h (WWR_WARP_SIZE and the
# wwrStream_t that columnwise_ell2_bridge.h also pulls in) plus the runtime
# libs; wwr.extension.parallel_for carries parallel_for.cuh and, transitively,
# the same wwr.device.
calaman_add_gpu_device_library(
  NAME
  calaman.columnwise_ell2.device
  SOURCES
  columnwise_ell2.cu
  LINK_PRIVATE
  calaman.reduce_columns
  wwr.extension.parallel_for
)

# The module itself: ordinary host C++23, no device code. instantiations.cpp is
# the explicit-instantiation implementation unit paired with interface.cppm's
# `extern template` list.
#
# INCLUDE_DIRS_PRIVATE is for columnwise_ell2_bridge.h, which interface.cppm
# includes in its global module fragment. LINK_PRIVATE wwr_backend is for the
# runtime.h that header pulls in there: a GMF cannot `import`, so
# wwrStream_t arrives by #include, and selected_backend.h behind it reads
# WWR_GPU_BACKEND_{CUDA,HIP} with no device pass to read __CUDACC__/__HIP__.
# LINK_PUBLIC wwr.runtime_api is the wwrStream_t the exported signature names.
calaman_add_cxx_module_library(
  NAME
  calaman.columnwise_ell2
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

# Force the device archive members in, or the explicit
# device::columnwise_ell2<T> instantiations are dropped (the module only names
# them through lazily-resolved calls) and a consumer fails to link -- the same
# whole-archive link lacpy and horner need. The install-interface spelling waits
# on the install tier
# (CALAMAN_INSTALL, docs/architecture.md §2); this build-interface link is what
# the tests need meanwhile.
target_link_libraries(
  calaman.columnwise_ell2
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.columnwise_ell2.device>"
)
