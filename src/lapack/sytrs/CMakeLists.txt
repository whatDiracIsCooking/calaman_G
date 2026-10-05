# calaman.sytrs -- LAPACK's ?sytrs, the solve of A X = B for a symmetric matrix
# already factored by the Bunch-Kaufman method (A = U D U^T or L D L^T). A host
# composition of the wrapped BLAS (ger/geru/gemv/swap) over the pivot sequence;
# its one non-BLAS stage -- applying the block-diagonal D^-1 to the right-hand
# sides -- is the device library. Complex is SYMMETRIC (no conjugation); the
# Hermitian cousin is calaman.hetrs.

# The device-kernel library. The two D^-1 applies ride
# wwr.extension.parallel_for (one thread per RHS column), so it links that
# INTERFACE library (carrying parallel_for.cuh + wwr.device + the bridge's
# runtime.h) PRIVATE, and calaman.common for the src/ root that resolves
# "common/elem_ops.cuh" -- the calaman.lacgv arrangement.
calaman_add_gpu_device_library(
  NAME
  calaman.sytrs.device
  SOURCES
  sytrs.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
  calaman.common
)

# All module links are PUBLIC because sytrs() is an exported template DEFINED in
# the interface (like pstrf): a consumer instantiates the body, so every module
# the body names must reach it. sytrs.cppm's GMF includes sytrs_bridge.h
# (INCLUDE_DIRS_PRIVATE, this directory), which pulls runtime.h; a GMF cannot
# import, so wwrStream_t arrives by #include and the backend define is read via
# LINK_PRIVATE wwr_backend -- exactly the calaman.pstrf arrangement.
#   wwr.wrappers.blas       ger, geru, gemv, swap
#   wwr.blas                handle / status / WWRBLAS_OP_* / wwrblasGetStream
#   wwr.runtime_api         wwrMemcpy(Async), wwrStreamSynchronize
#   wwr.complex             the two complex element types (instantiation list)
#   calaman.common          Uplo, usual_fp / real_fp, kOne / kNegativeOne
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.sytrs
  PRIMARY_INTERFACE
  sytrs.cppm
  LINK_PUBLIC
  wwr.wrappers.blas
  wwr.blas
  wwr.runtime_api
  wwr.complex
  calaman.common
  calaman.error_handling
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::sytrs_* kernel
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the whole-archive link lacpy needs.
target_link_libraries(
  calaman.sytrs PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.sytrs.device>"
)
