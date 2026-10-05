# calaman.lacgv -- conjugate a complex vector in place on the device, the GPU
# counterpart of LAPACK's ?lacgv (clacgv / zlacgv only; the conjugate of a real
# vector is itself). A shipped src/ module: written once against WarpWraps's
# `wwr*` names and built for either backend. Structurally the slimmest of the
# complex_cast family -- a single per-element functor launched through
# wwr.extension.parallel_for, conjugating through calaman.common's elem_ops.cuh.

# The device-kernel library. lacgv.cu launches its functor through
# wwr.extension.parallel_for, so it links that INTERFACE library PRIVATE, which
# carries parallel_for.cuh plus, transitively, wwr.device: complex.h for the
# neutral complex types and runtime.h (wwrStream_t, via lacgv_bridge.h), plus
# the backend runtime libs. It conjugates through calaman::device::elem_ops's
# conj, so it links calaman.common too, for the src/ root that resolves
# "common/elem_ops.cuh".
calaman_add_gpu_device_library(
  NAME
  calaman.lacgv.device
  SOURCES
  lacgv.cu
  LINK_PRIVATE
  wwr::extension::parallel_for
  calaman::common
)

# The module itself: ordinary host C++23, no device code. interface.cppm carries
# the exported wrapper; instantiations.cpp is the explicit-instantiation
# implementation unit paired with its `extern template` list.
#
# LINK_PUBLIC carries every module the exported surface names: wwr.runtime_api
# (the wwrStream_t it forwards), wwr.complex (the two complex element types),
# calaman.common (the complex_fp constraint), and calaman.error_handling
# (re-exported via `export import` for the Status return type; its src/ root
# resolves the "error_handling/error_macros.h" interface.cppm pulls into the GMF
# for CLM_TRY).
#
# LINK_PRIVATE wwr_backend is for the runtime.h that lacgv_bridge.h pulls into
# the global module fragment: a GMF cannot `import`, so wwrStream_t arrives by
# #include, and selected_backend.h behind it reads WWR_GPU_BACKEND_{CUDA,HIP}.
# INCLUDE_DIRS_PRIVATE is for lacgv_bridge.h itself (included from the GMF).
calaman_add_cxx_module_library(
  NAME
  calaman.lacgv
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr::runtime_api
  wwr::complex
  calaman::common
  calaman::error_handling
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::lacgv<T>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the same whole-archive link lacpy and
# complex_cast need. The install-interface spelling waits on the install tier
# (CALAMAN_INSTALL, docs/architecture.md §2); this build-interface link is what
# the tests need meanwhile.
calaman_link_device_archive(calaman.lacgv calaman.lacgv.device)
