# calaman.davidson -- block-Davidson eigensolver for the lowest n_roots
# eigenpairs of a symmetric operator known only through a matrix-vector-product
# callback. A shipped src/ module: ordinary host C++23, no device code of its
# own (the operator and preconditioner are caller callbacks), driving the
# wrapped BLAS and the wrapped symmetric eigensolver. Not a LAPACK routine, so
# it is its own module like calaman.feast.
#
# :buffer_size is the workspace layout/sizing; :solve is the iteration, both the
# standard (syevd) and the generalized/metric (sygvd) problem.
#
# The template surface is header-only (like the geqp3 family and feast),
# instantiated on use, so there is no instantiations.cpp.
#
# LINK_PUBLIC is the union of what the exported surface names: the runtime
# stream (wwr.runtime_api), the BLAS and solver handle/status/enum layers
# (wwr.blas, wwr.solver), the real_fp trait (calaman.common), the
# level-1/3 BLAS wrappers the iteration drives (wwr.wrappers.blas --
# gemm/axpy/dot/nrm2/scal) and the eigensolver wrappers (wwr.wrappers.solver --
# syevd and the *_bufferSize queries), the kOne/kZero/kNegativeOne constants and
# WorkspaceLayout/align_up carving (calaman.common), the ScopedPointerMode
# guard the solve forces host mode with (wwr.extension.blas), and
#   calaman.error_handling  Status, the cross-domain return type
#   calaman.iterative       IterationInfo, DavidsonInfo's base (re-exported)
#   calaman.ritz            classify_ritz (absolute form), ritz_rotate
# (every entry point returns it).
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
  calaman::ritz
  IMPORT_STD
)
