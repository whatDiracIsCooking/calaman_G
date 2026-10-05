# calaman.lahqr -- double-shift Francis QR for the Schur form of a small upper
# Hessenberg matrix, LAPACK's ?lahqr, and (optionally) accumulate the transform
# into Z. A shipped src/ module: written once against WarpWraps's `wwr*` names
# and built for either backend. The whole QR iteration runs in one single-thread
# kernel launched through wwr.extension.parallel_for -- structurally laexc's
# twin. The auxiliaries (?larfg/?lanv2 and the DROT it uses to finish a 2x2) are
# inlined in the .cu, so the module links no sibling LAPACK module.

# The device-kernel library. lahqr.cu launches its functor through
# wwr.extension.parallel_for, so it links that INTERFACE library PRIVATE, which
# carries parallel_for.cuh plus, transitively, wwr.device: runtime.h
# (wwrStream_t, via lahqr_bridge.h) and the backend runtime libs. The precision-
# neutral math (sqrt/fabs/hypot/copysign/pow/log) comes from wwr.wrappers.math's
# header, pulled in by #include in the .cu, so no extra link edge is needed.
calaman_add_gpu_device_library(
  NAME
  calaman.lahqr.device
  SOURCES
  lahqr.cu
  LINK_PRIVATE
  wwr::extension::parallel_for
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
# LINK_PRIVATE wwr_backend is for the runtime.h that lahqr_bridge.h pulls into
# the global module fragment: a GMF cannot `import`, so wwrStream_t arrives by
# #include, and selected_backend.h behind it reads WWR_GPU_BACKEND_{CUDA,HIP}.
# INCLUDE_DIRS_PRIVATE is for lahqr_bridge.h itself (included from the GMF).
calaman_add_cxx_module_library(
  NAME
  calaman.lahqr
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

# Force the device archive members in, or the explicit device::lahqr<T>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the whole-archive link laexc needs too.
calaman_link_device_archive(calaman.lahqr calaman.lahqr.device)
