# calaman.lantr -- the norm of a triangular (trapezoidal) matrix, the GPU
# counterpart of LAPACK's ?lantr.

# The device-kernel library: a per-column (per-row, for inf) partials kernel and
# a single-block fold, also calaman.lantp's and calaman.lantb's (one kernel,
# templated on the storage layout). tri_index carries the packed map;
# calaman.common carries the src/ root for common/block_reduce.cuh,
# common/elem_ops.cuh and common/enums.h plus, through it, wwr.device.
calaman_add_gpu_device_library(
  NAME
  calaman.lantr.device
  SOURCES
  lantr.cu
  LINK_PRIVATE
  calaman::tri_index
  calaman::common
)

# The module: host C++23 over the launcher declared in lantr_bridge.h, which
# interface.cppm includes in its GMF (hence INCLUDE_DIRS_PRIVATE, and
# wwr_backend PRIVATE for the selected_backend.h behind runtime.h there).
# PUBLIC links are what the exported surface names: wwrStream_t, MatrixNorm,
# Uplo and Diag (re-exported from calaman.common) and Status
# (calaman.error_handling, whose src/ root also resolves the GMF's
# error_macros.h for CLM_TRY); and wwr.complex, for the `extern template` list.
calaman_add_cxx_module_library(
  NAME
  calaman.lantr
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

# Force the device archive members in, or the explicit device::lantr<T, R>
# instantiations are dropped and a consumer fails to link -- as for lansb.
calaman_link_device_archive(calaman.lantr calaman.lantr.device)
