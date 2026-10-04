# calaman.trevc3 -- the left/right eigenvectors of a real quasi-triangular
# (Schur) matrix by back-substitution, LAPACK's ?trevc3 (HOWMNY = 'A',
# eigenvectors of T directly). A shipped src/ module: written once against
# WarpWraps's `wwr*` names and built for either backend. The per-block scaled
# solve is O(1) and each back-substitution AXPY/DOT touches O(N) entries, so it
# is one single-thread kernel launched through wwr.extension.parallel_for --
# structurally laqr5's twin. ?laln2 (the scaled 1x1/2x2 solve per block) is
# inlined in the .cu rather than imported: calaman.laln2 is a launch functor,
# unusable from inside a device kernel, so -- exactly as laexc inlines its
# auxiliaries -- the sweep links no sibling LAPACK module. Its inlined solve's
# complex divisions reach calaman.ladiv's header-only host/device ladiv_scalar
# root-relative ("lapack/ladiv/ladiv.h") off the src/ root, included not
# imported: a kernel reaches it by #include, not import.

# The device-kernel library. trevc3.cu launches its functor through
# wwr.extension.parallel_for, so it links that INTERFACE library PRIVATE, which
# carries parallel_for.cuh plus, transitively, wwr.device: runtime.h
# (wwrStream_t, via trevc3_bridge.h) and the backend runtime libs. The src/
# include root is added so "lapack/ladiv/ladiv.h" resolves root-relative --
# ladiv_scalar is header-only pure arithmetic, so there is nothing to LINK
# beyond it, the same hand-rolled root shape calaman.laln2 uses.
calaman_add_gpu_device_library(
  NAME
  calaman.trevc3.device
  SOURCES
  trevc3.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
)
target_include_directories(
  calaman.trevc3.device PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
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
# LINK_PRIVATE wwr_backend is for the runtime.h that trevc3_bridge.h pulls into
# the global module fragment: a GMF cannot `import`, so wwrStream_t arrives by
# #include, and selected_backend.h behind it reads WWR_GPU_BACKEND_{CUDA,HIP}.
# INCLUDE_DIRS_PRIVATE is for trevc3_bridge.h itself (included from the GMF).
calaman_add_cxx_module_library(
  NAME
  calaman.trevc3
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

# Force the device archive members in, or the explicit device::trevc3<T>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the whole-archive link laqr5 needs too.
target_link_libraries(
  calaman.trevc3 PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.trevc3.device>"
)
