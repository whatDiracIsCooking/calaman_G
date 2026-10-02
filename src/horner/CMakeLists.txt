# calaman.horner -- matrix polynomial evaluation p(A) = sum_k c_k A^k by
# Horner's method, in d gemms for degree d with a single n-by-n scratch block.
# A shipped src/ module, structurally the twin of calaman.lacpy: a host wrapper
# over wrapped BLAS plus a device kernel, written once against WarpWraps's
# `wwr*` names and built for either backend.
#
# Horner matrix-polynomial evaluation is not a LAPACK routine, so it is its own
# module rather than a calaman.linalg partition.

# The device-kernel library: the two diagonal updates of the recurrence
# (P := c_d I to seed it, P(i,i) += c_k to close each step). Both are
# hand-written functors launched through wwr.extension.parallel_for from
# horner.cu, so it links that INTERFACE library (which carries wwr.device + the
# bridge include root for the runtime.h that horner_bridge.h pulls in)
# PRIVATE, like calaman.linalg's downdate libraries and WarpWraps's own
# parallel_for consumers.
calaman_add_gpu_device_library(
  NAME
  calaman.horner.device
  SOURCES
  horner.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
)

# The module itself: ordinary host C++23, no device code. instantiations.cpp is
# the explicit-instantiation implementation unit paired with interface.cppm's
# `extern template` list.
#
# LINK_PUBLIC is the union of what the exported horner() body names: the gemm
# wrapper (wwr.wrappers.blas), the handle/status/op-constant/GetStream layer
# (wwr.blas), the wwrStream_t type the body hands to the launchers
# (wwr.runtime_api), calaman.common (kZero/kOne scalars, WorkspaceBuilder,
# all_nonnull) and calaman.error_handling (the Status return type, re-exported,
# and the CLM_TRY/CLM_REQUIRE macros its src/ include root publishes).
# LINK_PRIVATE wwr_backend is for the runtime.h that horner_bridge.h
# pulls into the global module fragment: a GMF cannot `import`, so wwrStream_t
# arrives by #include and selected_backend.h behind it reads WWR_GPU_BACKEND_*.
# INCLUDE_DIRS_PRIVATE is the source dir for horner_bridge.h itself.
calaman_add_cxx_module_library(
  NAME
  calaman.horner
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

# Force the device archive members in, or the explicit device::horner_*
# instantiations are dropped (the module only names them through
# lazily-resolved calls) and a consumer fails to link -- the same whole-archive
# link lacpy and calaman.linalg need. The install-interface spelling waits on
# the install tier (CALAMAN_INSTALL, docs/architecture.md §2).
target_link_libraries(
  calaman.horner
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.horner.device>"
)
