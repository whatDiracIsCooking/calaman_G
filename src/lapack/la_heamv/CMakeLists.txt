# calaman.la_heamv -- y := alpha * |A| * |x| + beta * |y| for a Hermitian A
# held in one triangle (LAPACK's ?la_heamv). It owns no device code: the kernel
# is calaman.la_syamv's, since CABS1 erases the Hermitian/symmetric difference.

# The module: host C++23 over the launcher declared in la_syamv_bridge.h, which
# interface.cppm includes in its GMF root-relative off the src/ root that
# calaman.common carries (wwr::backend PRIVATE for runtime.h there). PUBLIC
# links are what the exported surface names, as for calaman.la_syamv.
calaman_add_cxx_module_library(
  NAME
  calaman.la_heamv
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

# Force calaman.la_syamv's device archive in, or the device::la_syamv<T, R>
# instantiations are dropped and a consumer fails to link -- as for lanhe.
calaman_link_device_archive(calaman.la_heamv calaman.la_syamv.device)
