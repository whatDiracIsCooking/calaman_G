# calaman.laqr3 -- aggressive early deflation (AED) for the real nonsymmetric
# eigensolver, LAPACK's ?laqr3 (the recursive variant: the window Schur form
# comes from ?laqr4 when the window JW exceeds the ILAENV(12) crossover
# NMIN = 75, else from ?lahqr, exactly where ?laqr3 equals ?laqr2). A shipped
# src/ module: written once against WarpWraps's `wwr*` names, built for either
# backend.
#
# ?laqr4 is a HOST composition no device kernel can call, so unlike
# calaman.laqr2 (one single-thread kernel) this module is a host composition
# that SPLITS the AED pass around the window Schur form: the device half
# (laqr3.cu) is two launched kernels -- laqr3_setup (extract the window into T,
# V := I) and laqr3_finish (?laqr2's deflation / reorder / re-Hessenberg / slab
# tail) -- and interface.cppm sequences setup, calaman.laqr4, finish on the
# host. The JW <= NMIN regime, where ?laqr3 equals ?laqr2 exactly, is delegated
# whole to the calaman.laqr2 kernel. The finish kernel inlines the ?trexc /
# ?lanv2 / ?larfg / ?gehrd / ?ormhr auxiliaries (host modules unusable from a
# device kernel), exactly as laqr2 does.

# The device-kernel library. laqr3.cu launches its two functors through
# wwr.extension.parallel_for, so it links that INTERFACE library PRIVATE, which
# carries parallel_for.cuh plus, transitively, wwr.device: runtime.h
# (wwrStream_t, via laqr3_bridge.h) and the backend runtime libs. The precision-
# neutral math (sqrt/fabs/hypot/copysign/pow/log) comes from wwr.wrappers.math's
# header, pulled in by #include in the .cu, so no extra link edge is needed.
calaman_add_gpu_device_library(
  NAME
  calaman.laqr3.device
  SOURCES
  laqr3.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
)

# The module itself: ordinary host C++23, no device code. interface.cppm carries
# the exported wrapper; instantiations.cpp is the explicit-instantiation
# implementation unit paired with its `extern template` list.
#
# LINK_PUBLIC carries every module the exported surface names: wwr.runtime_api
# (the wwrStream_t it forwards), wwr.wrappers.common (the real_fp constraint),
# calaman.laqr2 (the delegated small-window pass) and calaman.laqr4 (the
# recursive window Schur form) -- both imported and instantiated in the exported
# body -- and calaman.error_handling (re-exported via `export import` for the
# Status return type; its src/ root resolves the "error_handling/error_macros.h"
# interface.cppm pulls into the GMF for CLM_TRY).
#
# LINK_PRIVATE wwr_backend is for the runtime.h that laqr3_bridge.h pulls into
# the global module fragment: a GMF cannot `import`, so wwrStream_t arrives by
# #include, and selected_backend.h behind it reads WWR_GPU_BACKEND_{CUDA,HIP}.
# INCLUDE_DIRS_PRIVATE is for laqr3_bridge.h itself (included from the GMF).
calaman_add_cxx_module_library(
  NAME
  calaman.laqr3
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr.runtime_api
  wwr.wrappers.common
  calaman.laqr2
  calaman.laqr4
  calaman.error_handling
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::laqr3_{setup,
# finish}<T> instantiations are dropped (the module only names them through
# lazily-resolved calls) and a consumer fails to link -- the whole-archive link
# laqr2 / laqr5 need too.
target_link_libraries(
  calaman.laqr3 PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.laqr3.device>"
)
