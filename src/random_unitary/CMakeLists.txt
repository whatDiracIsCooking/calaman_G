# calaman.random_unitary -- a random orthogonal / unitary matrix: fill n-by-n
# with standard-normal entries, then expand its Q factor. A shipped src/ module,
# written once against WarpWraps's `wwr*` names and built for either backend.
#
# Ordinary host C++23, no device code (no .cu, no bridge): the Gaussian fill is
# one call into wwr.extension.random_normal (which owns the per-backend kernel)
# and the QR is calaman.orthogonalize. instantiations.cpp is the
# explicit-instantiation unit paired with random_unitary.cppm's `extern
# template` list.
#
# All links are PUBLIC because the interface unit imports each as a named
# module: a consumer's build needs their BMIs to parse calaman.random_unitary.
#   calaman.orthogonalize        the QR that expands the explicit Q
#   wwr.extension.random_normal  the device-side standard-normal fill
#   wwr.solver          wwrsolverDnHandle_t / wwrsolverStatus_t / GetStream
#   wwr.runtime_api              wwrStream_t (for the fill's stream)
#   wwr.rand                     wwrrandState (the caller's generator states)
#   wwr.wrappers.common          wwrFloatComplex/wwrDoubleComplex
#   calaman.common               usual_fp (the element-type constraint)
#   calaman.error_handling       Status, the cross-domain return type
#                                (re-exported; its src/ root also resolves the
#                                CLM_TRY header "error_handling/error_macros.h")
calaman_add_cxx_module_library(
  NAME
  calaman.random_unitary
  PRIMARY_INTERFACE
  random_unitary.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  calaman.orthogonalize
  wwr.extension.random_normal
  wwr.solver
  wwr.runtime_api
  wwr.rand
  wwr.wrappers.common
  calaman.common
  calaman.error_handling
  IMPORT_STD
)
