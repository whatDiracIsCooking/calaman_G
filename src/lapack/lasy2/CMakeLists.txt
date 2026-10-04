# calaman.lasy2 -- solve the order-(1..2) Sylvester equation
# op(TL)*X + ISGN*X*op(TR) = SCALE*B on the device, the GPU counterpart of
# LAPACK's ?lasy2 (real s/d; LAPACK ships no complex variant). A shipped src/
# module: written once against WarpWraps's `wwr*` names and built for either
# backend. The whole solve is O(1) (X is at most 2x2), so it is one single
# thread launched through wwr.extension.parallel_for -- structurally lacgv's
# twin, minus the elem_ops.cuh header (real abs/max are inlined in the .cu).

# The device-kernel library. lasy2.cu launches its functor through
# wwr.extension.parallel_for, so it links that INTERFACE library PRIVATE, which
# carries parallel_for.cuh plus, transitively, wwr.device: runtime.h
# (wwrStream_t, via lasy2_bridge.h) and the backend runtime libs. No
# elem_ops.cuh: abs/max are real, so they are inline helpers in the .cu.
calaman_add_gpu_device_library(
  NAME
  calaman.lasy2.device
  SOURCES
  lasy2.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
)

# The module itself: ordinary host C++23, no device code. interface.cppm carries
# the exported wrapper; instantiations.cpp is the explicit-instantiation
# implementation unit paired with its `extern template` list.
#
# LINK_PUBLIC carries every module the exported surface names: wwr.runtime_api
# (the wwrStream_t it forwards), calaman.common (the real_fp constraint),
# and calaman.error_handling (re-exported via `export import` for the Status
# return type; its src/ root resolves the "error_handling/error_macros.h"
# interface.cppm pulls into the GMF for CLM_TRY).
#
# LINK_PRIVATE wwr_backend is for the runtime.h that lasy2_bridge.h pulls into
# the global module fragment: a GMF cannot `import`, so wwrStream_t arrives by
# #include, and selected_backend.h behind it reads WWR_GPU_BACKEND_{CUDA,HIP}.
# INCLUDE_DIRS_PRIVATE is for lasy2_bridge.h itself (included from the GMF).
calaman_add_cxx_module_library(
  NAME
  calaman.lasy2
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

# Force the device archive members in, or the explicit device::lasy2<T>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the whole-archive link lacgv needs too.
# The install-interface spelling waits on the install tier (CALAMAN_INSTALL,
# docs/architecture.md §2); this build-interface link is what the tests need
# meanwhile.
target_link_libraries(
  calaman.lasy2 PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.lasy2.device>"
)
