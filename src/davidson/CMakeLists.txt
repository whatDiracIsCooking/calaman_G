# calaman.davidson -- block-Davidson eigensolver for the lowest n_roots
# eigenpairs of a symmetric operator known only through a matrix-vector-product
# callback. A shipped src/ module: ordinary host C++23, no device code of its
# own (the operator and preconditioner are caller callbacks), driving the
# wrapped BLAS and the wrapped symmetric eigensolver. Not a LAPACK routine, so
# it is its own module like calaman.feast.
#
# SKELETON: :buffer_size is complete; :solve pins the interface and returns
# WWRBLAS_STATUS_NOT_SUPPORTED until the iteration lands in a follow-up PR.
#
# The template surface is header-only (like the geqp3 family and feast),
# instantiated on use, so there is no instantiations.cpp.
#
# LINK_PUBLIC is the union of what the exported surface names: the runtime
# stream (wwr.runtime_api), the BLAS and solver handle/status/enum layers
# (wwr.blas, wwr.solver), the real_fp trait (wwr.wrappers.common), the
# eigensolver buffer-size wrappers (wwr.wrappers.solver --
# syevd_bufferSize/sygvd_bufferSize), the WorkspaceLayout/align_up workspace
# carving (calaman.common), and
#   calaman.error_handling  Status, the cross-domain return type
# (every entry point returns it). The level-1/3 BLAS wrappers the iteration will
# call are deliberately NOT listed yet -- :solve names none of them -- and
# arrive with the implementation.
calaman_add_cxx_module_library(
  NAME
  calaman.davidson
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  buffer_size.cppm
  solve.cppm
  LINK_PUBLIC
  wwr.runtime_api
  wwr.blas
  wwr.solver
  wwr.wrappers.common
  wwr.wrappers.solver
  calaman.common
  calaman.error_handling
  IMPORT_STD
)
