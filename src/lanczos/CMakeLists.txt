# calaman.lanczos -- thick-restart Lanczos for the extreme eigenpairs of a real
# symmetric operator given as a single-vector matvec callback. Not a LAPACK
# routine, so it is its own module, like calaman.davidson.

# The device-kernel library behind lanczos_bridge.h: the step write/normalise,
# the breakdown guard and the restart arrowhead. Raw <<<>>> launches plus
# common/block_reduce.cuh, so it links wwr.device and calaman.common (for the
# src/ root) PRIVATE, like calaman.feast.device.
calaman_add_gpu_device_library(
  NAME
  calaman.lanczos.device
  SOURCES
  lanczos.cu
  LINK_PRIVATE
  wwr::device
  calaman::common
)

# The module: :types holds the options/result/which/callback; :buffer_size the
# workspace; :ritz the Ritz extraction; :solve the driver. Header-only
# templates, so no instantiations.cpp.
#
# LINK_PUBLIC is what the exported surface names: the stream (wwr.runtime_api),
# the BLAS/solver handles and enums (wwr.blas, wwr.solver), wwrrandState
# (wwr.rand), gemm/gemv/nrm2 (wwr.wrappers.blas), syevd
# (wwr.wrappers.solver), ScopedPointerMode (wwr.extension.blas), the random
# start (wwr.extension.init_state, wwr.extension.random_normal), real_fp and
# the workspace carving (calaman.common), Status (calaman.error_handling), and
# IterationInfo, LanczosInfo's base (calaman.iterative, re-exported).
#
# LINK_PRIVATE wwr::backend and INCLUDE_DIRS_PRIVATE are for lanczos_bridge.h,
# which :buffer_size, :ritz and :solve include in their GMF: a GMF cannot
# import, so wwrStream_t arrives via runtime.h, whose backend define
# wwr::backend supplies.
calaman_add_cxx_module_library(
  NAME
  calaman.lanczos
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  types.cppm
  buffer_size.cppm
  ritz.cppm
  solve.cppm
  LINK_PUBLIC
  wwr::runtime_api
  wwr::blas
  wwr::solver
  wwr::rand
  wwr::wrappers::blas
  wwr::wrappers::solver
  wwr::extension::blas
  wwr::extension::init_state
  wwr::extension::random_normal
  calaman::common
  calaman::error_handling
  calaman::iterative
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive in, or the explicit device::lanczos_* instantiations
# are dropped (the module names them only through lazily-resolved calls) -- the
# whole-archive link feast and nnls need.
calaman_link_device_archive(calaman.lanczos calaman.lanczos.device)
