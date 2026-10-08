# calaman.davidson -- block-Davidson eigensolver for the lowest n_roots
# eigenpairs of a symmetric linear_operator (or a block sigma callback). A
# shipped src/ module: ordinary host C++23, no device code of its own (the
# operator and preconditioner are the caller's), driving the wrapped BLAS and
# the wrapped symmetric eigensolver. Not a LAPACK routine, so it is its own
# module like calaman.feast.
#
# :buffer_size is the workspace layout/sizing; :solve is the iteration, both the
# standard (syevd) and the generalized/metric (sygvd) problem. Header-only
# templates, instantiated on use, so there is no instantiations.cpp.
#
# LINK_PUBLIC is the union of what the exported surface names: the runtime
# stream (wwr.runtime_api), the BLAS and solver handle/status/enum layers
# (wwr.blas, wwr.solver), the level-1/3 BLAS wrappers the iteration drives
# (wwr.wrappers.blas -- gemm/axpy/dot/nrm2/scal) and the eigensolver wrappers
# (wwr.wrappers.solver -- syevd/sygvd and the *_bufferSize queries), real_fp,
# kOne/kZero/kNegativeOne and WorkspaceLayout carving (calaman.common), the
# ScopedPointerMode guard the solve forces host mode with (wwr.extension.blas),
#   calaman.error_handling   Status, the cross-domain return type
#   calaman.iterative        IterationInfo, DavidsonInfo's base (re-exported)
#   calaman.linear_operator  linear_operator, what davidson_solve applies
#   calaman.ritz             classify_ritz, ritz_rotate
calaman_add_cxx_module_library(
  NAME
  calaman.davidson
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  buffer_size.cppm
  solve.cppm
  LINK_PUBLIC
  wwr::runtime_api
  wwr::blas
  wwr::solver
  wwr::wrappers::blas
  wwr::wrappers::solver
  wwr::extension::blas
  calaman::common
  calaman::error_handling
  calaman::iterative
  calaman::linear_operator
  calaman::ritz
  IMPORT_STD
)
