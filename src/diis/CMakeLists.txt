# calaman.diis -- Pulay DIIS extrapolation over a device-resident ring of
# (vector, residual) pairs: two gemv per push for the incremental Gram, a
# one-block bordered-solve kernel, one gemv for the extrapolation.

# The device-kernel library: the hand-launched bordered solve. wwr.device
# carries runtime.h (wwrStream_t, WWR_WARP_SIZE) and calaman.common the
# math.cuh root.
calaman_add_gpu_device_library(
  NAME
  calaman.diis.device
  SOURCES
  diis.cu
  LINK_PRIVATE
  wwr::device
  calaman::common
)

# PUBLIC: the types the exported signatures name -- wwrblasHandle_t (wwr.blas),
# wwrStream_t (wwr.runtime_api), real_fp (calaman.common) and Status
# (calaman.error_handling, re-exported). wwr.wrappers.blas and
# wwr.extension.blas are named only by the template bodies, but those bodies are
# exported, so an importer needs them too. wwr_backend is for the runtime.h
# diis_bridge.h pulls into the GMF.
calaman_add_cxx_module_library(
  NAME
  calaman.diis
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr::blas
  wwr::runtime_api
  wwr::wrappers::blas
  wwr::extension::blas
  calaman::common
  calaman::error_handling
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::diis_solve<T>
# instantiations are dropped and a consumer fails to link.
calaman_link_device_archive(calaman.diis calaman.diis.device)
