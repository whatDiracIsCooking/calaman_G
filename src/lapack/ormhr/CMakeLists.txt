# calaman.ormhr -- LAPACK's ?ormhr: multiply a general matrix by the orthogonal
# Q from ?gehrd (Q*C, Q^T*C, C*Q, C*Q^T). A pure host index shift over
# wwr::ormqr -- the gehrd reflectors are the geqrf layout shifted down one row,
# so ormhr forwards the nh = ihi-ilo reflectors at A(ilo+1, ilo) with tau(ilo)
# to one ormqr over the shifted C submatrix. Real only (float/double; unmqr the
# complex counterpart). No .cu of its own -- ormqr carries the device work.
#
# All links are PUBLIC because ormhr() is an exported template DEFINED in the
# interface (like geqp3): a consumer instantiates the body, so every module the
# body names must reach it.
#   wwr.wrappers.solver  ormqr, ormqr_bufferSize
#   wwr.blas             wwrblasSideMode_t, wwrblasOperation_t, SIDE/OP enums
#   wwr.solver           wwrsolverDnHandle_t / wwrsolverStatus_t / WWRSOLVER_*
#   wwr.runtime_api      wwrErrorInvalidValue -- the illegal-argument Status
#   calaman.common       Side, Trans
#   calaman.error_handling  Status, the cross-domain return type (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.ormhr
  PRIMARY_INTERFACE
  ormhr.cppm
  LINK_PUBLIC
  wwr::wrappers::solver
  wwr::blas
  wwr::solver
  wwr::runtime_api
  calaman::common
  calaman::error_handling
  IMPORT_STD
)
