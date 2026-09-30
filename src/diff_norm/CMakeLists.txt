# calaman.diff_norm -- a chosen norm (ell_1 / ell_2 / ell_inf) of the difference
# y - x, as one axpy (y := -x + y) followed by one BLAS reduction. Pure host
# C++23: axpy, asum, nrm2 and iamax are stock BLAS, so this reaches the
# outermost WarpWraps layer (wwr.wrappers.blas) and owns no device code -- no
# .cu, no bridge header, no explicit-instantiation unit.
#
# All links are PUBLIC because diff_norm() is an exported template DEFINED in
# the interface (no extern-template firewall, unlike calaman.lacpy): a consumer
# instantiates the body itself, so every module the body names must reach it.
#   wwr.wrappers.blas       axpy, asum, nrm2, iamax
#   wwr.blas                wwrblasHandle_t / wwrblasStatus_t / WWRBLAS_STATUS_*
#   wwr.runtime_api         wwrMemcpy + wwrSuccess for the ell_inf read
#   calaman.common.constants kNegativeOne<T>, the fixed axpy scalar
calaman_add_cxx_module_library(
  NAME
  calaman.diff_norm
  PRIMARY_INTERFACE
  diff_norm.cppm
  LINK_PUBLIC
  wwr.wrappers.blas
  wwr.blas
  wwr.runtime_api
  calaman.common.constants
)
