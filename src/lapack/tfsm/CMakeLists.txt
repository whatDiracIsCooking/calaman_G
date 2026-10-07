# calaman.tfsm -- LAPACK's ?tfsm, the triangular solve with the triangular
# operand in RFP storage. A host composition: two wrapped trsm calls and one
# gemm on the sub-blocks calaman.rfp_blocks places, so it owns no device code.
#
# All links are PUBLIC because tfsm() is an exported template DEFINED in the
# interface (like calaman.hfrk): a consumer instantiates the body itself, so
# every module the body names must reach it.
#   wwr.wrappers.blas       gemm, trsm
#   wwr.blas                handle, wwrblasGetStream, BLAS enums, statuses
#   wwr.runtime_api         wwrStream_t, wwrMemset2DAsync
#   wwr.complex             the complex element types
#   calaman.common          Trans, Uplo, Side, Diag, usual_fp
#   calaman.rfp_blocks      the RFP sub-block geometry
#   calaman.error_handling  Status, the cross-domain return type
calaman_add_cxx_module_library(
  NAME
  calaman.tfsm
  PRIMARY_INTERFACE
  tfsm.cppm
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
