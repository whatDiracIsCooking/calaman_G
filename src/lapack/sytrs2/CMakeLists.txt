# calaman.sytrs2 -- LAPACK's ?sytrs2, the level-3 solve of A X = B for a
# symmetric matrix already Bunch-Kaufman factored. Same result as calaman.sytrs,
# but it pulls the interchanges to the ends so the triangular solves are two
# trsm calls. A host composition of wrapped BLAS-3 (trsm) + host-driven swaps
# over its own device stages: the ?syconv VALUE move and the D^-1 block applies.
# Complex is SYMMETRIC (no conjugation); the Hermitian cousin is calaman.hetrs2.

# The device-kernel library. syconv_value is hand-launched; the two D^-1 applies
# ride wwr.extension.parallel_for. It links that INTERFACE library (carrying
# parallel_for.cuh + wwr.device + the bridge's runtime.h) PRIVATE, and
# calaman.common for the src/ root that resolves "common/elem_ops.cuh" and
# "common/constants.h" -- the calaman.sytrs arrangement.
calaman_add_gpu_device_library(
  NAME
  calaman.sytrs2.device
  SOURCES
  sytrs2.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
  calaman.common
)

# All module links are PUBLIC because sytrs2()/sytrs2_bufferSize() are exported
# templates DEFINED in the interface (like sytrs): a consumer instantiates the
# body, so every module the body names must reach it. sytrs2.cppm's GMF includes
# sytrs2_bridge.h (INCLUDE_DIRS_PRIVATE, this dir), which pulls runtime.h; a GMF
# cannot import, so wwrStream_t arrives by #include and the backend define read
# via LINK_PRIVATE wwr_backend -- the calaman.sytrs arrangement.
#   wwr.wrappers.blas       trsm, swap
#   wwr.blas                handle, status, WWRBLAS OP/SIDE/FILL/DIAG enums
#   wwr.runtime_api         wwrMemcpy(Async), wwrStreamSynchronize
#   wwr.complex             the two complex element types (instantiation list)
#   calaman.common          Uplo, usual_fp, kOne
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.sytrs2
  PRIMARY_INTERFACE
  sytrs2.cppm
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

# Force the device archive members in, or the explicit device::sytrs2_* /
# syconv_* kernel instantiations are dropped (the module only names them through
# lazily-resolved calls) and a consumer fails to link -- the whole-archive link.
target_link_libraries(
  calaman.sytrs2 PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.sytrs2.device>"
)
