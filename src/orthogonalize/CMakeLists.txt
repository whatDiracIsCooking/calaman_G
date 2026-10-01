# calaman.orthogonalize -- an orthonormal basis for a matrix's columns via QR:
# geqrf in place, then orgqr/ungqr to expand the thin Q. A shipped src/ module,
# written once against WarpWraps's `wwr*` names and built for either backend.
#
# Ordinary host C++23, no device code (no .cu, no bridge): geqrf/orgqr/ungqr are
# stock dense-solver calls, so this reaches the outermost WarpWraps layer
# (wwr.wrappers.solver). instantiations.cpp is the explicit-instantiation
# implementation unit paired with orthogonalize.cppm's `extern template` list.
#
# All links are PUBLIC because the interface unit imports each as a named
# module: a consumer's build needs their BMIs to parse calaman.orthogonalize.
#   wwr.wrappers.solver  geqrf/orgqr/ungqr (+ bufferSize), usual_fp, complex ts
#   wwr.solver           wwrsolverDnHandle_t / wwrsolverStatus_t / the status
#   calaman.common       align_up (the 256-element tau rounding)
calaman_add_cxx_module_library(
  NAME
  calaman.orthogonalize
  PRIMARY_INTERFACE
  orthogonalize.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr.wrappers.solver
  wwr.solver
  calaman.common
  IMPORT_STD
)
