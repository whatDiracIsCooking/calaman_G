# calaman.laexc -- swap two adjacent 1x1 / 2x2 diagonal blocks of a real Schur
# form by an orthogonal similarity, LAPACK's ?laexc, and (optionally) accumulate
# the transform into Q. A shipped src/ module: written once against WarpWraps's
# `wwr*` names and built for either backend. The block arithmetic is O(1) but
# the rotations/reflectors touch O(N) rows/cols, so it is one single-thread
# kernel launched through wwr.extension.parallel_for -- structurally lasy2's
# twin. The auxiliaries (?lartg/?lasy2/?lanv2/?larfg and the reflector apply)
# are inlined in the .cu, so the module links no sibling LAPACK module.

# The device-kernel library. laexc.cu launches its functor through
# wwr.extension.parallel_for, so it links that INTERFACE library PRIVATE, which
# carries parallel_for.cuh plus, transitively, wwr.device: runtime.h
# (wwrStream_t, via laexc_bridge.h) and the backend runtime libs. The precision-
# neutral math (sqrt/fabs/hypot/copysign/pow/log) comes from wwr.wrappers.math's
# header, pulled in by #include in the .cu, so no extra link edge is needed.
calaman_add_gpu_device_library(
  NAME
  calaman.laexc.device
  SOURCES
  laexc.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
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
# LINK_PRIVATE wwr_backend is for the runtime.h that laexc_bridge.h pulls into
# the global module fragment: a GMF cannot `import`, so wwrStream_t arrives by
# #include, and selected_backend.h behind it reads WWR_GPU_BACKEND_{CUDA,HIP}.
# INCLUDE_DIRS_PRIVATE is for laexc_bridge.h itself (included from the GMF).
calaman_add_cxx_module_library(
  NAME
  calaman.laexc
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

# Force the device archive members in, or the explicit device::laexc<T>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the whole-archive link lasy2 needs too.
target_link_libraries(
  calaman.laexc PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.laexc.device>"
)
