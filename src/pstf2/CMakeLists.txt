# calaman.pstf2 -- LAPACK's ?pstf2, the unblocked level-2 pivoted Cholesky panel
# of a symmetric positive semidefinite matrix. A host composition of the wrapped
# BLAS swap / scal / gemv; its two non-BLAS stages -- the fused diagonal-and-
# pivot reduction and zeroing the trailing factor -- are the device library.

# The device-kernel library. The pivot reduction is hand-launched; the
# trailing-factor zeroing rides wwr.extension.parallel_for, so it links that
# INTERFACE library (carrying wwr.device + the bridge include root) PRIVATE,
# like calaman.laqp2 / calaman.laqps / calaman.nnls.
calaman_add_gpu_device_library(
  NAME
  calaman.pstf2.device
  SOURCES
  pstf2.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
)

# All module links are PUBLIC because pstf2() is an exported template DEFINED in
# the interface (like calaman.laqp2): a consumer instantiates the body, so every
# module the body names must reach it. pstf2.cppm's GMF includes pstf2_bridge.h
# (INCLUDE_DIRS_PRIVATE, this directory), which pulls runtime.h; a GMF cannot
# import, so wwrStream_t arrives by #include and the backend define is read via
# LINK_PRIVATE wwr_backend -- exactly the calaman.laqp2 arrangement.
#   wwr.wrappers.blas       swap, scal, gemv
#   wwr.blas                handle / status / WWRBLAS_OP_* / WWRBLAS_STATUS_*
#   wwr.runtime_api         wwrMemcpy(Async), wwrMemsetAsync, wwrStreamSync
#   calaman.common          Uplo (the referenced-triangle selector)
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.pstf2
  PRIMARY_INTERFACE
  pstf2.cppm
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

# Force the device archive members in, or the explicit device::pstf2_* kernel
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the whole-archive link lacpy needs.
target_link_libraries(
  calaman.pstf2
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.pstf2.device>"
)
