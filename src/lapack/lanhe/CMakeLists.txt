# calaman.lanhe -- the norm of a complex Hermitian matrix from the one triangle
# `uplo` names, the GPU counterpart of LAPACK's ?lanhe. It owns no device code:
# the kernel is calaman.lansy's, run with its Hermitian flag set.

# The module: host C++23 over the launcher declared in lansy_bridge.h, which
# interface.cppm includes in its GMF root-relative off the src/ root that
# calaman.common carries (wwr_backend PRIVATE for the selected_backend.h behind
# runtime.h there). PUBLIC links are what the exported surface names:
# wwrStream_t, the complex element types, MatrixNorm and Uplo (re-exported from
# calaman.common) and Status (calaman.error_handling, whose src/ root also
# resolves the GMF's error_macros.h for CLM_TRY).
calaman_add_cxx_module_library(
  NAME
  calaman.lanhe
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr::runtime_api
  wwr::complex
  calaman::common
  calaman::error_handling
  LINK_PRIVATE
  wwr::backend
  IMPORT_STD
)

# Force calaman.lansy's device archive in, or the device::lansy<T, R>
# instantiations are dropped and a consumer fails to link -- as for lansy.
calaman_link_device_archive(calaman.lanhe calaman.lansy.device)
