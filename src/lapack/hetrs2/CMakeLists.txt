# calaman.hetrs2 -- LAPACK's ?hetrs2, the level-3 solve of A X = B for a
# Hermitian matrix already Bunch-Kaufman factored. Same result as calaman.hetrs,
# but it pulls the interchanges to the ends so the triangular solves are two
# trsm calls (the second conjugate-transpose). A host composition of wrapped
# BLAS-3 (trsm) + host-driven swaps over its own ?syconv VALUE move and
# Hermitian D^-1 kernels. Complex only (c/z); real Hermitian is symmetric
# (calaman.sytrs2).

# The device-kernel library. syconv_value is hand-launched; the two D^-1 applies
# ride wwr.extension.parallel_for. It links that INTERFACE library (carrying
# parallel_for.cuh + wwr.device + the bridge's runtime.h) PRIVATE, and
# calaman.common for the src/ root that resolves "common/elem_ops.cuh" and
# "common/constants.h" -- the calaman.hetrs arrangement.
calaman_add_gpu_device_library(
  NAME
  calaman.hetrs2.device
  SOURCES
  hetrs2.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
  calaman.common
)

# All module links are PUBLIC because hetrs2()/hetrs2_bufferSize() are exported
# templates DEFINED in the interface (like hetrs): a consumer instantiates the
# body, so every module the body names must reach it. hetrs2.cppm's GMF includes
# hetrs2_bridge.h (INCLUDE_DIRS_PRIVATE, this dir), which pulls runtime.h; a GMF
# cannot import, so wwrStream_t arrives by #include and the backend define is
# read via LINK_PRIVATE wwr_backend -- the calaman.hetrs arrangement.
#   wwr.wrappers.blas       trsm, swap
#   wwr.blas                handle, status, OP/SIDE/FILL/DIAG enums, GetStream
#   wwr.runtime_api         wwrMemcpy(Async), wwrStreamSynchronize
#   wwr.complex             the two complex element types (instantiation list)
#   calaman.common          Uplo, complex_fp, kOne
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.hetrs2
  PRIMARY_INTERFACE
  hetrs2.cppm
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

# Force the device archive members in, or the explicit device::hetrs2_* kernel
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the whole-archive link.
target_link_libraries(
  calaman.hetrs2 PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.hetrs2.device>"
)
