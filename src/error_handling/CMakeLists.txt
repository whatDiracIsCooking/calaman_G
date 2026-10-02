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
#
# INCLUDE_DIRS_PUBLIC is the src/ root, exported so a consumer can pull the
# CLM_TRY macro in via `#include "error_handling/error_macros.h"` from a global
# module fragment. error_macros.h is a GMF-shared header (a macro cannot ride
# an `import`), the same shape calaman.common publishes for "common/enums.h".
# It is wrapped in $<BUILD_INTERFACE:> so the raw build path does not leak into
# the install interface, which waits on the install tier (CALAMAN_INSTALL,
# docs/architecture.md §2).
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
  INCLUDE_DIRS_PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
