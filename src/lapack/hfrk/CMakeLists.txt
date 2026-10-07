# calaman.hfrk -- LAPACK's ?hfrk (c/z) and ?sfrk (s/d), the rank-k update of a
# Hermitian/symmetric matrix in RFP storage. A host composition: two wrapped
# herk/syrk calls and one gemm on the sub-blocks calaman.rfp_blocks places, so
# it owns no device code.
#
# All links are PUBLIC because hfrk() is an exported template DEFINED in the
# interface (like calaman.larcm): a consumer instantiates the body itself, so
# every module the body names must reach it.
#   wwr.wrappers.blas       gemm, herk, syrk
#   wwr.blas                handle, wwrblasGetStream, op/fill enums, statuses
#   wwr.runtime_api         wwrStream_t, wwrMemsetAsync
#   wwr.complex             the complex element types
#   calaman.common          Trans, Uplo, usual_fp, ComplexToRealType
#   calaman.rfp_blocks      the RFP sub-block geometry
#   calaman.error_handling  Status, the cross-domain return type
calaman_add_cxx_module_library(
  NAME
  calaman.hfrk
  PRIMARY_INTERFACE
  hfrk.cppm
  LINK_PUBLIC
  wwr::wrappers::blas
  wwr::blas
  wwr::runtime_api
  wwr::complex
  calaman::common
  calaman::rfp_blocks
  calaman::error_handling
  IMPORT_STD
)
