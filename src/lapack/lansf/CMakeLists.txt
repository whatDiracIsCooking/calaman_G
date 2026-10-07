# calaman.lansf -- the norm of a real symmetric matrix in Rectangular Full
# Packed storage, the GPU counterpart of LAPACK's ?lansf.

# The device-kernel library: a per-column partials kernel and a single-block
# fold, also calaman.lanhf's (one kernel, with a Hermitian flag). tri_index
# carries the RFP map (rfp_index); calaman.common carries the src/ root for
# common/block_reduce.cuh, common/elem_ops.cuh and common/enums.h plus, through
# it, wwr.device (runtime.h, complex.h, the runtime libs).
calaman_add_gpu_device_library(
  NAME
  calaman.lansf.device
  SOURCES
  lansf.cu
  LINK_PRIVATE
  calaman::tri_index
  calaman::common
)

# The module: host C++23 over the launcher declared in lansf_bridge.h, which
# interface.cppm includes in its GMF (hence INCLUDE_DIRS_PRIVATE, and
# wwr_backend PRIVATE for the selected_backend.h behind runtime.h there).
# PUBLIC links are what the exported surface names: wwrStream_t, MatrixNorm,
# Trans and Uplo (re-exported from calaman.common) and Status
# (calaman.error_handling, whose src/ root also resolves the GMF's
# error_macros.h for CLM_TRY).
calaman_add_cxx_module_library(
  NAME
  calaman.lansf
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

# Force the device archive members in, or the explicit device::lansf<T, R>
# instantiations are dropped and a consumer fails to link -- as for lansy.
calaman_link_device_archive(calaman.lansf calaman.lansf.device)
