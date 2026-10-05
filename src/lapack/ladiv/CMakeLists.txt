# calaman.ladiv -- real-arithmetic complex division by Smith's algorithm,
# LAPACK's ?ladiv (float/double only; the complex cladiv/zladiv take a
# different, complex-input signature). The scalar helper laln2 calls per-thread
# is header-only (ladiv.h, CLM_HOST_DEVICE); this module also batches n of them
# on the device, structurally calaman.lartg's sibling -- a single per-element
# functor through wwr.extension.parallel_for -- so the oracle exercises it on a
# card.

# The device-kernel library. ladiv.cu launches its functor through
# wwr.extension.parallel_for, so it links that INTERFACE library PRIVATE, which
# carries parallel_for.cuh plus, transitively, wwr.device and runtime.h
# (wwrStream_t, via ladiv_bridge.h). The per-element arithmetic is the
# header-only ladiv_scalar from ladiv.h, included bare (same directory), so no
# extra include root and no wwr.math link: Smith's algorithm uses only
# compare/multiply/divide.
calaman_add_gpu_device_library(
  NAME
  calaman.ladiv.device
  SOURCES
  ladiv.cu
  LINK_PRIVATE
  wwr::extension::parallel_for
)

# The module itself: ordinary host C++23, no device code. interface.cppm carries
# the exported batched driver; instantiations.cpp is the explicit-instantiation
# implementation unit paired with its `extern template` list.
#
# LINK_PUBLIC carries every module the exported surface names: wwr.runtime_api
# (the wwrStream_t it forwards) and calaman.error_handling (re-exported via
# `export import` for the Status return type; its src/ root resolves the
# "error_handling/error_macros.h" the GMF pulls in for CLM_TRY).
#
# LINK_PRIVATE wwr_backend is for the runtime.h that ladiv_bridge.h pulls into
# the global module fragment: a GMF cannot `import`, so wwrStream_t arrives by
# #include, and selected_backend.h behind it reads WWR_GPU_BACKEND_{CUDA,HIP}.
# INCLUDE_DIRS_PRIVATE is for ladiv_bridge.h itself (included from the GMF).
calaman_add_cxx_module_library(
  NAME
  calaman.ladiv
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr::runtime_api
  calaman::error_handling
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::ladiv<T>
# instantiations are dropped (the module only names them through
# lazily-resolved calls) and a consumer fails to link -- the same whole-archive
# link lartg needs. The install-interface spelling waits on the install tier
# (CALAMAN_INSTALL, docs/architecture.md §2); this build-interface link is what
# the tests need meanwhile.
calaman_link_device_archive(calaman.ladiv calaman.ladiv.device)
