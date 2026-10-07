# calaman.lacrm -- LAPACK's ?lacrm, C := A * B for a complex A and a real
# square B. A host composition: calaman.complex_cast's split_planes /
# merge_planes around two wrapped real gemms, so it owns no device code.
#
# All links are PUBLIC because lacrm() is an exported template DEFINED in the
# interface (like calaman.larfb): a consumer instantiates the body itself, so
# every module the body names must reach it.
#   wwr.wrappers.blas       gemm
#   wwr.blas                handle, wwrblasGetStream, WWRBLAS_OP_N, statuses
#   wwr.runtime_api         wwrStream_t, wwrGetLastError
#   wwr.complex             the complex element types
#   calaman.common          complex_fp, ComplexToRealType, carve_workspace
#   calaman.complex_cast    split_planes, merge_planes
#   calaman.error_handling  Status, the cross-domain return type
calaman_add_cxx_module_library(
  NAME
  calaman.lacrm
  PRIMARY_INTERFACE
  lacrm.cppm
  LINK_PUBLIC
  wwr::wrappers::blas
  wwr::blas
  wwr::runtime_api
  wwr::complex
  calaman::common
  calaman::complex_cast
  calaman::error_handling
  IMPORT_STD
)
