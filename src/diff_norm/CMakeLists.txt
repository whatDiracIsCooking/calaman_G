# calaman.diff_norm -- a chosen norm (ell_1 / ell_2 / ell_inf) of the difference
# y - x, as one axpy (y := -x + y) followed by one BLAS reduction. Pure host
# C++23: axpy, asum, nrm2 and iamax are stock BLAS, so this reaches the
# outermost WarpWraps layer (wwr.wrappers.blas) and owns no device code -- no
# .cu, no bridge header, no explicit-instantiation unit (the ell_inf device path
# calls calaman.set_element, whose kernel lives there).
#
# All links are PUBLIC because diff_norm() is an exported template DEFINED in
# the interface (no extern-template firewall, unlike calaman.lacpy): a consumer
# instantiates the body itself, so every module the body names must reach it.
# calaman.error_handling is additionally re-exported (export import) because it
# supplies the Status return type, whose member functions a consumer must see.
#   wwr.wrappers.blas       axpy, asum, nrm2, iamax
#   wwr.blas                handle/WWRBLAS_STATUS_*, pointer mode, GetStream
#   wwr.runtime_api         wwrMemcpyAsync/StreamSynchronize (host ell_inf read)
#                           and wwrMallocAsync/FreeAsync (device ell_inf index)
#   wwr.extension.blas      ScopedPointerMode -- forces host mode for the axpy
#   calaman.common          kNegativeOne<T>, the fixed axpy scalar (:constants)
#   calaman.error_handling  Status, the cross-domain return type
#   calaman.set_element     set_element_abs -- ell_inf's device-mode magnitude
calaman_add_cxx_module_library(
  NAME
  calaman.diff_norm
  PRIMARY_INTERFACE
  diff_norm.cppm
  LINK_PUBLIC
  wwr::wrappers::blas
  wwr::blas
  wwr::runtime_api
  wwr::extension::blas
  calaman::common
  calaman::error_handling
  calaman::set_element
)
