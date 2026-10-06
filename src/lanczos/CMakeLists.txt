# calaman.lanczos -- thick-restart Lanczos for the extreme eigenpairs of a real
# symmetric operator given as a single-vector matvec callback. Not a LAPACK
# routine, so it is its own module, like calaman.davidson.
#
# :types holds the options/result/which/callback; :buffer_size the workspace;
# :solve the driver. Header-only templates, so there is no instantiations.cpp.
# The device-kernel library declared in lanczos_bridge.h is not built yet.
#
# LINK_PUBLIC is what the exported surface names: the stream (wwr.runtime_api),
# the BLAS/solver handles and enums (wwr.blas, wwr.solver), wwrrandState
# (wwr.rand), syevd_bufferSize (wwr.wrappers.solver), real_fp and the workspace
# carving (calaman.common), and Status (calaman.error_handling).
#
# LINK_PRIVATE wwr::backend and INCLUDE_DIRS_PRIVATE are for lanczos_bridge.h,
# which :buffer_size includes in its GMF: a GMF cannot import, so wwrStream_t
# arrives via runtime.h, whose backend define wwr::backend supplies.
calaman_add_cxx_module_library(
  NAME
  calaman.lanczos
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  types.cppm
  buffer_size.cppm
  solve.cppm
  LINK_PUBLIC
  wwr::runtime_api
  wwr::blas
  wwr::solver
  wwr::rand
  wwr::wrappers::solver
  calaman::common
  calaman::error_handling
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)
