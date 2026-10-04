# calaman.set_element -- read one element of a device array by a device-held
# index (gather, or its magnitude). A shipped src/ module: written once against
# WarpWraps's `wwr*` names and built for either backend.
#
# Not a LAPACK routine -- it is the device-pointer-mode back half of I?AMAX:
# iamax writes a 1-based index to the device, and no vendor BLAS call reads that
# element back, so calaman.diff_norm's ell_inf path needs this to work in device
# pointer mode. Structurally the twin of calaman.complex_cast's kernel half:
# per-element functors launched through wwr.extension.parallel_for, taking |.|
# through calaman.common's elem_ops.cuh modulus (never .x/.y, which hipComplex
# lacks).

# The device-kernel library. set_element.cu launches its functor through
# wwr.extension.parallel_for, so it links that INTERFACE library PRIVATE, which
# carries parallel_for.cuh plus, transitively, wwr.device: runtime.h
# (wwrStream_t, pulled in by set_element_bridge.h) and the backend runtime libs.
# It takes the magnitude through calaman::device::elem_ops::modulus, so it links
# calaman.common too, for the src/ root that resolves "common/elem_ops.cuh" (and
# the complex.h behind it, via wwr.device).
calaman_add_gpu_device_library(
  NAME
  calaman.set_element.device
  SOURCES
  set_element.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
  calaman.common
)

# The module itself: ordinary host C++23, no device code. interface.cppm carries
# the two exported wrappers; instantiations.cpp is the explicit-instantiation
# implementation unit paired with its `extern template` list.
#
# LINK_PUBLIC carries every module the exported surface names, so an importer
# sees them: wwr.runtime_api (the wwrStream_t each forwards), wwr.complex (the
# two complex element types in the extern-template list) and wwr.wrappers.common
# (the usual_fp constraint and the ComplexToRealType the magnitude output is
# spelled with).
#
# LINK_PRIVATE wwr_backend is for the runtime.h that set_element_bridge.h pulls
# into the global module fragment: a GMF cannot `import`, so wwrStream_t arrives
# by #include, and selected_backend.h behind it reads
# WWR_GPU_BACKEND_{CUDA,HIP}. INCLUDE_DIRS_PRIVATE is for set_element_bridge.h
# itself (included from the GMF).
calaman_add_cxx_module_library(
  NAME
  calaman.set_element
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr.runtime_api
  wwr.complex
  wwr.wrappers.common
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::set_element*
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the same whole-archive link
# complex_cast and laset need. The install-interface spelling waits on the
# install tier (CALAMAN_INSTALL, docs/architecture.md §2); this build-interface
# link is what the tests need meanwhile.
target_link_libraries(
  calaman.set_element
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.set_element.device>"
)
