# calaman.rscl -- scale a strided vector by the reciprocal of a real scalar
# without over/underflow, the GPU counterpart of LAPACK's ?rscl (the real-scalar
# family: srscl, drscl, csrscl, zdrscl). A shipped src/ module: written once
# against WarpWraps's `wwr*` names, built for either backend.

# The device-kernel library. rscl.cu launches its functor through
# wwr.extension.parallel_for, so it links that INTERFACE library PRIVATE, which
# carries parallel_for.cuh plus, transitively, wwr.device: complex.h for the
# neutral complex types and runtime.h (wwrStream_t, via rscl_bridge.h), plus the
# backend runtime libs. It scales through calaman::device::elem_ops, so it links
# calaman.common too, for the src/ root that resolves "common/elem_ops.cuh".
calaman_add_gpu_device_library(
  NAME
  calaman.rscl.device
  SOURCES
  rscl.cu
  LINK_PRIVATE
  wwr::extension::parallel_for
  calaman::common
)

# The module itself: ordinary host C++23, no device code -- the guarded
# multiplier loop is host scalar arithmetic in the interface, so IMPORT_STD
# supplies the <limits>/<cmath> names it uses.
#
# INCLUDE_DIRS_PRIVATE is for rscl_bridge.h, which interface.cppm includes in
# its global module fragment; LINK_PRIVATE wwr_backend is for the runtime.h that
# header pulls in there (a GMF cannot `import`, and selected_backend.h behind it
# reads WWR_GPU_BACKEND_{CUDA,HIP}). PUBLIC links are what the exported surface
# names: wwrStream_t, the complex element types, ComplexToRealType
# (calaman.common) and Status (calaman.error_handling, re-exported through
# `export import`; its src/ root also resolves the
# "error_handling/error_macros.h" the GMF pulls in for CLM_TRY).
calaman_add_cxx_module_library(
  NAME
  calaman.rscl
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

# Force the device archive members in, or the explicit device::rscl<T, R>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the whole-archive link lacgv needs too.
calaman_link_device_archive(calaman.rscl calaman.rscl.device)
