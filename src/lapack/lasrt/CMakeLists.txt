# calaman.lasrt -- sort a 1-D array into increasing or decreasing order on the
# device, the GPU counterpart of LAPACK's ?lasrt. A shipped src/ module: written
# once against WarpWraps's `wwr*` names and built for either backend.

# The device-kernel library. The sort is one hand-launched single-thread kernel
# (quicksort reverting to insertion sort, an explicit stack), so it links the
# pieces directly rather than riding wwr.extension.parallel_for: wwr.device
# carries WarpWraps's src include root (runtime.h, which lasrt_bridge.h and
# lasrt.cu pull in for wwrStream_t) plus the runtime libs; calaman.common
# exports the src/ root that resolves "common/enums.h" (SortDir, shared across
# the host/device boundary). The link is PRIVATE, as every device link is.
calaman_add_gpu_device_library(
  NAME
  calaman.lasrt.device
  SOURCES
  lasrt.cu
  LINK_PRIVATE
  wwr::device
  calaman::common
)

# The module itself: ordinary host C++23, no device code. interface.cppm carries
# the exported API and the thin front-door wrapper; instantiations.cpp is the
# explicit-instantiation implementation unit paired with its `extern template`
# list.
#
# INCLUDE_DIRS_PRIVATE is for lasrt_bridge.h, which interface.cppm includes in
# its global module fragment. LINK_PRIVATE wwr_backend is for the runtime.h that
# header pulls in there: a GMF cannot `import`, so wwrStream_t arrives by
# #include, and selected_backend.h behind it reads WWR_GPU_BACKEND_{CUDA,HIP}
# with no device pass to read __CUDACC__/__HIP__.
#
# calaman.common is PUBLIC: the interface imports it for SortDir (:enums) and
# re-exports that name, so a consumer of calaman.lasrt reaches SortDir through
# it. It also carries the src/ root that resolves the "common/enums.h" that
# lasrt_bridge.h pulls into the global module fragment.
# calaman.error_handling is also PUBLIC: the interface re-exports it (export
# import) for the Status return type, and its src/ root resolves the
# "error_handling/error_macros.h" interface.cppm pulls into the GMF for
# CLM_TRY / CLM_REQUIRE.
calaman_add_cxx_module_library(
  NAME
  calaman.lasrt
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr::runtime_api
  calaman::common
  calaman::error_handling
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::lasrt<T>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link. The install-interface spelling waits on
# the install tier (CALAMAN_INSTALL, docs/architecture.md §2); this build-
# interface link is what the tests need meanwhile.
calaman_link_device_archive(calaman.lasrt calaman.lasrt.device)
