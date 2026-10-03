# calaman.lartg -- generate plane rotations on the device, the batched lift of
# LAPACK's ?lartg (float/double only; the complex clartg/zlartg differ
# materially, like calaman.larfg's). A shipped src/ module: written once against
# WarpWraps's `wwr*` names and built for either backend. Structurally lacgv's
# sibling -- a single per-element functor launched through
# wwr.extension.parallel_for -- but computing a rotation, not conjugating.

# The device-kernel library. lartg.cu launches its functor through
# wwr.extension.parallel_for, so it links that INTERFACE library PRIVATE, which
# carries parallel_for.cuh plus, transitively, wwr.device -- the include root
# where wrappers/math/math.cuh (wwr::fabs / sqrt / copysign / fmin / fmax) and
# runtime.h (wwrStream_t, via lartg_bridge.h) resolve, plus the backend runtime
# libs. There is no elem_ops link: the kernel is real-only and touches no
# complex arithmetic.
calaman_add_gpu_device_library(
  NAME
  calaman.lartg.device
  SOURCES
  lartg.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
)

# The module itself: ordinary host C++23, no device code. interface.cppm carries
# the exported wrapper; instantiations.cpp is the explicit-instantiation
# implementation unit paired with its `extern template` list.
#
# LINK_PUBLIC carries every module the exported surface names: wwr.runtime_api
# (the wwrStream_t it forwards) and calaman.error_handling (re-exported via
# `export import` for the Status return type; its src/ root resolves the
# "error_handling/error_macros.h" interface.cppm pulls into the GMF for CLM_TRY).
#
# LINK_PRIVATE wwr_backend is for the runtime.h that lartg_bridge.h pulls into
# the global module fragment: a GMF cannot `import`, so wwrStream_t arrives by
# #include, and selected_backend.h behind it reads WWR_GPU_BACKEND_{CUDA,HIP}.
# INCLUDE_DIRS_PRIVATE is for lartg_bridge.h itself (included from the GMF).
calaman_add_cxx_module_library(
  NAME
  calaman.lartg
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr.runtime_api
  calaman.error_handling
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::lartg<T>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the same whole-archive link lacgv and
# lascl2 need. The install-interface spelling waits on the install tier
# (CALAMAN_INSTALL, docs/architecture.md §2); this build-interface link is what
# the tests need meanwhile.
target_link_libraries(
  calaman.lartg
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.lartg.device>"
)
