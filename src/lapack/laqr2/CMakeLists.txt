# calaman.laqr2 -- aggressive early deflation (AED) for the real nonsymmetric
# eigensolver, LAPACK's ?laqr2 (the small-window variant that forms the window
# Schur form with ?lahqr, not the recursive ?laqr4 of ?laqr3). A shipped src/
# module: written once against WarpWraps's `wwr*` names and built for either
# backend. The window work is O(JW^3) and each slab update O(N*JW), so it is one
# single-thread kernel launched through wwr.extension.parallel_for -- laqr5's
# twin. The window auxiliaries ?lahqr, ?trexc (over ?laexc), ?lanv2, ?larfg,
# ?gehrd and ?ormhr are inlined in the .cu rather than imported: those sibling
# modules are HOST compositions that stage through device memcpies, unusable
# from inside a device kernel, so -- exactly as laqr5 inlines its ?laqr1/?larfg
# -- the sweep links no sibling LAPACK module.

# The device-kernel library. laqr2.cu launches its functor through
# wwr.extension.parallel_for, so it links that INTERFACE library PRIVATE, which
# carries parallel_for.cuh plus, transitively, wwr.device: runtime.h
# (wwrStream_t, via laqr2_bridge.h) and the backend runtime libs. The precision-
# neutral math (sqrt/fabs/hypot/copysign/pow/log) comes from wwr.wrappers.math's
# header, pulled in by #include in the .cu, so no extra link edge is needed.
calaman_add_gpu_device_library(
  NAME
  calaman.laqr2.device
  SOURCES
  laqr2.cu
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
# LINK_PRIVATE wwr_backend is for the runtime.h that laqr2_bridge.h pulls into
# the global module fragment: a GMF cannot `import`, so wwrStream_t arrives by
# #include, and selected_backend.h behind it reads WWR_GPU_BACKEND_{CUDA,HIP}.
# INCLUDE_DIRS_PRIVATE is for laqr2_bridge.h itself (included from the GMF).
calaman_add_cxx_module_library(
  NAME
  calaman.laqr2
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

# Force the device archive members in, or the explicit device::laqr2<T>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the whole-archive link laqr5 needs too.
calaman_link_device_archive(calaman.laqr2 calaman.laqr2.device)
