# calaman.error_handling -- Status, the one return type the three WarpWraps
# error domains (wwrblasStatus_t, wwrsolverStatus_t, wwrError_t) cast into. A
# pure-host value type: no device code, no .cu, no implementation unit -- the
# whole struct is defined inline in the interface, so a consumer that calls
# ok()/name()/message() names every module the bodies touch.
#
# All links are PUBLIC for that reason (as calaman.diff_norm is): the interface
# imports them and consumers inherit the dependency transitively.
#   wwr.runtime_api       wwrError_t, the runtime domain's type
#   wwr.extension.common  the error_type traits + wwrError_t's specialisation
#   wwr.extension.blas    wwrblasStatus_t (re-exports wwr.blas) + its trait
#   wwr.extension.solver  wwrsolverStatus_t (re-exports wwr.solver) + its trait
#
# The blas/solver trait specialisations live in the wwr.extension.* modules, not
# in wwr.blas/wwr.solver -- which is why those extension modules are the
# dependency rather than the bare neutral layer: importing wwr.blas alone would
# leave success_code<wwrblasStatus_t>() unspecialised.
calaman_add_cxx_module_library(
  NAME
  calaman.error_handling
  PRIMARY_INTERFACE
  error_handling.cppm
  LINK_PUBLIC
  wwr.runtime_api
  wwr.extension.common
  wwr.extension.blas
  wwr.extension.solver
)
