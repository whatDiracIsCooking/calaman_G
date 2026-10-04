# calaman.laln2 -- solve the scaled 1x1 or 2x2 real system
# (ca*op(A) - w*D) X = SCALE*B with a real or complex right-hand side, the GPU
# counterpart of LAPACK's ?laln2 (real element type; w may be complex). A
# shipped src/ module: written once against WarpWraps's `wwr*` names and built
# for either backend. The whole solve is O(1) (X is at most 2x2), so it is one
# single thread launched through wwr.extension.parallel_for -- structurally
# lasy2's twin. The complex divisions reach the header-only, host/device
# ladiv_scalar from calaman.ladiv's ladiv.h (issue #94 depends on #86), included
# root-relative off the src/ root rather than imported: a kernel reaches it by
# #include, not import.

# The device-kernel library. laln2.cu launches its functor through
# wwr.extension.parallel_for, so it links that INTERFACE library PRIVATE, which
# carries parallel_for.cuh plus, transitively, wwr.device: runtime.h
# (wwrStream_t, via laln2_bridge.h) and the backend runtime libs. The src/
# include root is added so "lapack/ladiv/ladiv.h" resolves root-relative --
# ladiv_scalar is header-only pure arithmetic (compare/multiply/divide), so
# there is nothing to LINK beyond it, the same hand-rolled root shape
# calaman.reduce_columns uses.
calaman_add_gpu_device_library(
  NAME
  calaman.laln2.device
  SOURCES
  laln2.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
)
target_include_directories(
  calaman.laln2.device PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

# The module itself: ordinary host C++23, no device code. interface.cppm carries
# the exported wrapper; instantiations.cpp is the explicit-instantiation
# implementation unit paired with its `extern template` list.
#
# LINK_PUBLIC carries every module the exported surface names: wwr.runtime_api
# (the wwrStream_t it forwards), wwr.wrappers.common (the real_fp constraint),
# and calaman.error_handling (re-exported via `export import` for the Status
# return type; its src/ root resolves the "error_handling/error_macros.h"
# interface.cppm pulls into the GMF for CLM_TRY).
#
# LINK_PRIVATE wwr_backend is for the runtime.h that laln2_bridge.h pulls into
# the global module fragment: a GMF cannot `import`, so wwrStream_t arrives by
# #include, and selected_backend.h behind it reads WWR_GPU_BACKEND_{CUDA,HIP}.
# INCLUDE_DIRS_PRIVATE is for laln2_bridge.h itself (included from the GMF).
calaman_add_cxx_module_library(
  NAME
  calaman.laln2
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr.runtime_api
  wwr.wrappers.common
  calaman.error_handling
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::laln2<T>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the whole-archive link lasy2 needs too.
# The install-interface spelling waits on the install tier (CALAMAN_INSTALL,
# docs/architecture.md §2); this build-interface link is what the tests need
# meanwhile.
target_link_libraries(
  calaman.laln2 PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.laln2.device>"
)
