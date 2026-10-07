# calaman.ritz -- the Ritz-pair bookkeeping lanczos, davidson and feast share:
# ritz_select, classify_ritz, ritz_rotate. Header-only templates over real_fp,
# so no instantiations.cpp and no device code.
#
# LINK_PUBLIC is what the interface unit imports: wwrblasHandle_t and the enums
# (wwr.blas), gemm (wwr.wrappers.blas), ScopedPointerMode (wwr.extension.blas),
# kOne/kZero/real_fp (calaman.common), and Status/PointerModeStatus
# (calaman.error_handling, re-exported; its src/ root also resolves the CLM_TRY
# header the GMF includes).
calaman_add_cxx_module_library(
  NAME
  calaman.ritz
  PRIMARY_INTERFACE
  ritz.cppm
  LINK_PUBLIC
  wwr::blas
  wwr::wrappers::blas
  wwr::extension::blas
  calaman::common
  calaman::error_handling
  IMPORT_STD
)
