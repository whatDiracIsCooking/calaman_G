# calaman.shift_invert -- DenseShiftInvert, the linear_operator of
# (A - sigma I)^{-1} for a dense real symmetric A: factor once by Bunch-Kaufman
# (wwr::sytrf), solve per apply (calaman.sytrs2). Its own module, not a lanczos
# partition, so any linear_operator consumer can take it. Pure host, no device
# code: the kernels arrive, with their whole-archive links, through lacpy, laset
# and sytrs2.
#
# All links are PUBLIC: the surface is header-only templates a consumer
# instantiates, so every module the bodies name must reach it.
#   wwr.runtime_api         wwrStream_t, wwrMemcpyAsync, wwrStreamSynchronize
#   wwr.blas / wwr.solver   the two handle types, WWRBLAS_* enums and statuses
#   wwr.wrappers.blas       axpy (the diagonal shift)
#   wwr.wrappers.solver     sytrf, sytrf_bufferSize
#   wwr.extension.blas      ScopedPointerMode
#   calaman.common          real_fp, Uplo, Region, kOne, carve_workspace
#   calaman.lacpy           A's triangle into the factor region
#   calaman.laset           the -sigma vector
#   calaman.sytrs2          the solve against the factor
#   calaman.linear_operator the concept the model satisfies (re-exported)
#   calaman.error_handling  Status, devinfo_verdict (re-exported)
calaman_add_cxx_module_library(
  NAME
  calaman.shift_invert
  PRIMARY_INTERFACE
  shift_invert.cppm
  LINK_PUBLIC
  wwr::runtime_api
  wwr::blas
  wwr::solver
  wwr::wrappers::blas
  wwr::wrappers::solver
  wwr::extension::blas
  calaman::common
  calaman::lacpy
  calaman::laset
  calaman::sytrs2
  calaman::linear_operator
  calaman::error_handling
  IMPORT_STD
)
