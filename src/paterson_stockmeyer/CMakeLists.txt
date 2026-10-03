# calaman.paterson_stockmeyer -- matrix polynomial p(A) = sum_k c_k A^k by the
# Paterson-Stockmeyer scheme, in O(sqrt(d)) gemms for degree d by blocking the
# terms into a polynomial in X = A^s. A shipped src/ module, the
# asymptotically-cheaper twin of calaman.horner: a host wrapper over wrapped
# BLAS plus a device kernel, written once against WarpWraps's `wwr*` names and
# built for either backend.
#
# Paterson-Stockmeyer matrix-polynomial evaluation is not a LAPACK routine, so
# it is its own module rather than a calaman.linalg partition.

# The device-kernel library: fusing each block B_j from the precomputed powers
# of A in a single pass (one hand-written functor launched through
# wwr.extension.parallel_for from paterson_stockmeyer.cu), so it links that
# INTERFACE library (which carries wwr.device + the bridge include root for the
# runtime.h that paterson_stockmeyer_bridge.h pulls in) PRIVATE, like
# calaman.horner's device library and WarpWraps's own parallel_for consumers.
calaman_add_gpu_device_library(
  NAME
  calaman.paterson_stockmeyer.device
  SOURCES
  paterson_stockmeyer.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
)

# The module itself: ordinary host C++23, no device code. instantiations.cpp is
# the explicit-instantiation implementation unit paired with interface.cppm's
# `extern template` list.
#
# LINK_PUBLIC is the union of what the exported paterson_stockmeyer() body
# names: the gemm wrapper (wwr.wrappers.blas), the handle/status/op/GetStream
# layer (wwr.blas), the wwrStream_t type the body hands to the launcher
# (wwr.runtime_api) and calaman.common (kZero/kOne scalars, WorkspaceBuilder,
# align_up). calaman.error_handling is additionally re-exported (export import)
# because it supplies the Status return type, whose member functions a consumer
# must see.
#   calaman.error_handling  Status, the cross-domain return type
# LINK_PRIVATE wwr_backend is for the runtime.h that
# paterson_stockmeyer_bridge.h pulls into the global module fragment: a GMF
# cannot `import`, so wwrStream_t arrives by #include and selected_backend.h
# behind it reads WWR_GPU_BACKEND_*. INCLUDE_DIRS_PRIVATE is the source dir for
# paterson_stockmeyer_bridge.h itself.
calaman_add_cxx_module_library(
  NAME
  calaman.paterson_stockmeyer
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr.wrappers.blas
  wwr.blas
  wwr.runtime_api
  calaman.common
  calaman.error_handling
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit
# device::paterson_stockmeyer_build_block instantiations are dropped (the module
# only names them through lazily-resolved calls) and a consumer fails to link --
# the same whole-archive link calaman.horner and calaman.laqps need. The
# install-interface spelling waits on the install tier (CALAMAN_INSTALL,
# docs/architecture.md §2).
target_link_libraries(
  calaman.paterson_stockmeyer
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.paterson_stockmeyer.device>"
)
