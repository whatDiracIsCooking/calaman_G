# calaman.lange -- the norm of a general column-major matrix, the GPU
# counterpart of LAPACK's ?lange. A shipped src/ module: written once against
# WarpWraps's `wwr*` names and built for either backend.

# The device-kernel library. lange.cu owns only the per-row abs-sum kernel (the
# infinity norm's stage one); everything else defers to two headers, so it links
# both, PRIVATE as every device-library link is:
#   calaman.reduce_columns      the segmented column reduce (both stages)
#   wwr.extension.parallel_for  the one-element sqrt (the Frobenius norm's end)
# calaman.reduce_columns carries the src/ include root for its header plus its
# own device deps -- wwr.device, which provides runtime.h (WWR_WARP_SIZE and the
# wwrStream_t that lange_bridge.h also pulls in) plus the runtime libs;
# wwr.extension.parallel_for carries parallel_for.cuh and, transitively, that
# same wwr.device. calaman.common carries the src/ root that resolves
# "common/enums.h" (MatrixNorm), which lange_bridge.h and lange.cu #include.
calaman_add_gpu_device_library(
  NAME
  calaman.lange.device
  SOURCES
  lange.cu
  LINK_PRIVATE
  calaman.reduce_columns
  wwr.extension.parallel_for
  calaman.common
)

# The module itself: ordinary host C++23, no device code. instantiations.cpp is
# the explicit-instantiation implementation unit paired with interface.cppm's
# `extern template` list.
#
# INCLUDE_DIRS_PRIVATE is for lange_bridge.h, which interface.cppm includes in
# its global module fragment. LINK_PRIVATE wwr_backend is for the runtime.h that
# header pulls in there: a GMF cannot `import`, so wwrStream_t arrives by
# #include, and selected_backend.h behind it reads WWR_GPU_BACKEND_{CUDA,HIP}
# with no device pass to read __CUDACC__/__HIP__.
#
# calaman.common is PUBLIC: the interface imports it for MatrixNorm (:enums) and
# re-exports that name, so a consumer of calaman.lange reaches MatrixNorm via
# it; it also carries the src/ root that resolves the "common/enums.h"
# lange_bridge.h pulls into the GMF. calaman.error_handling is also PUBLIC: the
# interface re-exports it (export import) for the Status return type, and its
# src/ root resolves the "error_handling/error_macros.h" interface.cppm pulls
# into the GMF for CLM_TRY. LINK_PUBLIC wwr.runtime_api is the wwrStream_t the
# exported signature names.
calaman_add_cxx_module_library(
  NAME
  calaman.lange
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr.runtime_api
  calaman.common
  calaman.error_handling
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::lange<T>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the same whole-archive link lacpy and
# the columnwise_* modules need. The install-interface spelling waits on the
# install tier (CALAMAN_INSTALL, docs/architecture.md §2); this build-interface
# link is what the tests need meanwhile.
target_link_libraries(
  calaman.lange
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.lange.device>"
)
