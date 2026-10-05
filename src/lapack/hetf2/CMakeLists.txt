# calaman.hetf2 -- LAPACK's ?hetf2, the unblocked Hermitian Bunch-Kaufman
# factorization (A = U D U^H or L D L^H) of a Hermitian indefinite matrix. The
# from-scratch factorization the vendors omit for Hermitian (they ship only the
# symmetric ?sytrf). A host-driven column loop runs the Bunch-Kaufman pivot
# search (wwr::iamax + per-step scalar reads) and the 1x1 update (her/scal); its
# two non-BLAS stages -- the Hermitian pivot interchange and the 2x2 rank-2
# update -- are the device library. Complex only (c/z); real Hermitian is
# symmetric (?sytrf).

# The device-kernel library. The interchange is hand-launched (single thread);
# the rank-2 update rides wwr.extension.parallel_for. It links that INTERFACE
# library (carrying parallel_for.cuh + wwr.device + the bridge's runtime.h)
# PRIVATE, and calaman.common for the src/ root that resolves
# "common/elem_ops.cuh" -- the calaman.hetrs arrangement.
calaman_add_gpu_device_library(
  NAME
  calaman.hetf2.device
  SOURCES
  hetf2.cu
  LINK_PRIVATE
  wwr::extension::parallel_for
  calaman::common
)

# All module links are PUBLIC because hetf2()/hetf2_bufferSize() are exported
# templates DEFINED in the interface (like hetrs): a consumer instantiates the
# body, so every module the body names must reach it. hetf2.cppm's GMF includes
# hetf2_bridge.h (INCLUDE_DIRS_PRIVATE, this dir), which pulls runtime.h; a GMF
# cannot import, so wwrStream_t arrives by #include and the backend define is
# read via LINK_PRIVATE wwr_backend -- the calaman.hetrs arrangement.
#   wwr.wrappers.blas       iamax, her, scal
#   wwr.blas                handle, status, FILL_MODE enums, wwrblasGetStream
#   wwr.runtime_api         wwrMemcpy(Async), wwrStreamSynchronize
#   wwr.complex             the two complex element types (instantiation list)
#   calaman.common          Uplo, complex_fp, ComplexToRealType
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.hetf2
  PRIMARY_INTERFACE
  hetf2.cppm
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

# Force the device archive members in, or the explicit device::hetf2_* kernel
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the whole-archive link.
calaman_link_device_archive(calaman.hetf2 calaman.hetf2.device)
