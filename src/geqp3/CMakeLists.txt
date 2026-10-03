# calaman.geqp3 -- LAPACK's ?geqp3, the top-level QR-with-column-pivoting
# driver: the all-free BLOCKED path (seeds vn1/vn2 with one wwr::nrm2 per
# column, then calaman.laqps blocks finishing with a calaman.laqp2 tail above a
# small crossover, else one calaman.laqp2 panel) and the fixed-prefix path
# (pre-permute the fixed columns, wwr::geqrf the leading block, wwr::ormqr Q^T
# onto the free tail, then the all-free path on the free trailing submatrix).
# Both paths are pure host compositions over already-wrapped routines; geqp3 has
# no .cu of its own (calaman.laqps / calaman.laqp2 carry the downdate kernels,
# and their device archives arrive transitively through those PUBLIC links).
#
# All links are PUBLIC because geqp3() is an exported template DEFINED in the
# interface (like calaman.diff_norm): a consumer instantiates the body, so every
# module the body names must reach it. The fixed-prefix overload adds the two
# solver links.
#   wwr.wrappers.blas    nrm2, wwrblasSideMode_t, wwrblasOperation_t
#   wwr.wrappers.solver  geqrf, geqrf_bufferSize, ormqr, ormqr_bufferSize
#   wwr.blas             wwrblasHandle_t / wwrblasStatus_t / WWRBLAS_STATUS_*
#   wwr.solver           wwrsolverDnHandle_t / wwrsolverStatus_t / WWRSOLVER_*
#   wwr.runtime_api      wwrMemcpy(Async), wwrStreamSynchronize, wwrSuccess
#   calaman.laqp2        calaman::laqp2
#   calaman.laqps        calaman::laqps
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.geqp3
  PRIMARY_INTERFACE
  geqp3.cppm
  LINK_PUBLIC
  wwr.wrappers.blas
  wwr.wrappers.solver
  wwr.blas
  wwr.solver
  wwr.runtime_api
  calaman.laqp2
  calaman.laqps
  calaman.error_handling
  IMPORT_STD
)
