# calaman.shifted_cocg -- multi-shift block COCG for (z_e I - A) X_e = B, A real
# symmetric and known only as a linear_operator. Not a LAPACK routine, so it is
# its own module, like calaman.lanczos.

# The device-kernel library behind shifted_cocg_bridge.h: the per-column Lanczos
# step and the per-(shift, column) COCG recurrences. Raw <<<>>> launches plus
# common/block_reduce.cuh, so it links wwr.device and calaman.common (for the
# src/ root) PRIVATE, like calaman.lanczos.device.
calaman_add_gpu_device_library(
  NAME
  calaman.shifted_cocg.device
  SOURCES
  shifted_cocg.cu
  LINK_PRIVATE
  wwr::device
  calaman::common
)

# The module: one header-only template interface, so no instantiations.cpp.
#
# LINK_PUBLIC is what the exported surface names: the stream and the
# memcpy/memset/sync (wwr.runtime_api), the WWRBLAS_STATUS_* codes (wwr.blas),
# real_fp and the workspace carving (calaman.common), Status
# (calaman.error_handling), IterationInfo (calaman.iterative) and
# linear_operator (calaman.linear_operator), the last three re-exported.
#
# LINK_PRIVATE wwr::backend and INCLUDE_DIRS_PRIVATE are for
# shifted_cocg_bridge.h in the GMF: a GMF cannot import, so wwrStream_t arrives
# via runtime.h, whose backend define wwr::backend supplies.
calaman_add_cxx_module_library(
  NAME
  calaman.shifted_cocg
  PRIMARY_INTERFACE
  shifted_cocg.cppm
  LINK_PUBLIC
  wwr::runtime_api
  wwr::blas
  calaman::common
  calaman::error_handling
  calaman::iterative
  calaman::linear_operator
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive in, or the explicit device::cocg_* instantiations
# are dropped (the module names them only through lazily-resolved calls) -- the
# whole-archive link lanczos and feast need.
calaman_link_device_archive(calaman.shifted_cocg calaman.shifted_cocg.device)
