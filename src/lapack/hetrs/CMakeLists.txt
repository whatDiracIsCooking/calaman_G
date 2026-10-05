# calaman.hetrs -- LAPACK's ?hetrs, the solve of A X = B for a Hermitian matrix
# already Bunch-Kaufman factored (A = U D U^H or L D L^H). A host walk of the
# pivot sequence over the wrapped BLAS (geru/gemv/swap) plus its own device
# stages: the real-reciprocal 1x1 apply, the Hermitian 2x2 apply, and the row
# conjugation that brackets each conjugate-transpose gemv. Complex only (c/z); a
# real Hermitian matrix is symmetric, which is calaman.sytrs.

# The device-kernel library. The three stages ride wwr.extension.parallel_for,
# so it links that INTERFACE library (carrying parallel_for.cuh + wwr.device +
# the bridge's runtime.h) PRIVATE, and calaman.common for the src/ root that
# resolves "common/elem_ops.cuh" -- the calaman.sytrs arrangement.
calaman_add_gpu_device_library(
  NAME
  calaman.hetrs.device
  SOURCES
  hetrs.cu
  LINK_PRIVATE
  wwr::extension::parallel_for
  calaman::common
)

# All module links are PUBLIC because hetrs() is an exported template DEFINED in
# the interface (like sytrs): a consumer instantiates the body, so every module
# the body names must reach it. hetrs.cppm's GMF includes hetrs_bridge.h
# (INCLUDE_DIRS_PRIVATE, this dir), which pulls runtime.h; a GMF cannot import,
# so wwrStream_t arrives by #include and the backend define is read via
# LINK_PRIVATE wwr_backend -- the calaman.sytrs arrangement.
#   wwr.wrappers.blas       geru, gemv, swap
#   wwr.blas                handle / status / WWRBLAS_OP_* / wwrblasGetStream
#   wwr.runtime_api         wwrMemcpy(Async), wwrStreamSynchronize
#   wwr.complex             the two complex element types (instantiation list)
#   calaman.common          Uplo, complex_fp, kOne / kNegativeOne
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.hetrs
  PRIMARY_INTERFACE
  hetrs.cppm
  LINK_PUBLIC
  wwr::wrappers::blas
  wwr::blas
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

# Force the device archive members in, or the explicit device::hetrs_* kernel
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the whole-archive link.
calaman_link_device_archive(calaman.hetrs calaman.hetrs.device)
