# calaman.lanht -- the norm of a complex Hermitian tridiagonal (d, e), the GPU
# counterpart of LAPACK's ?lanht. It owns no device code: the kernel is
# calaman.lanst's, instantiated for a complex off-diagonal.

# The module: host C++23 over the launcher declared in lanst_bridge.h, which
# interface.cppm includes in its GMF root-relative off the src/ root that
# calaman.common carries (wwr_backend PRIVATE for the selected_backend.h behind
# runtime.h there). PUBLIC links are what the exported surface names:
# wwrStream_t, the complex element types, MatrixNorm (re-exported from
# calaman.common) and Status (calaman.error_handling).
calaman_add_cxx_module_library(
  NAME
  calaman.lanht
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

# Force calaman.lanst's device archive in, or the device::lanst<T, R>
# instantiations are dropped and a consumer fails to link -- as for lanst.
calaman_link_device_archive(calaman.lanht calaman.lanst.device)
