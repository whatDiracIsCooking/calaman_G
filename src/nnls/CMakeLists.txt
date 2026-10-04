# calaman.nnls -- non-negative least squares (Lawson-Hanson active set). A
# host composition of wrapped BLAS (gemv/trsv/nrm2), calaman.geqp3 for the
# passive-set pivoted QR and wwr::ormqr to apply Q; its device stage is the P/Z
# active-set bookkeeping -- the masked argmax, min-ratio step test, compaction,
# the passive-column gather and the rank off the pivoted R diagonal -- below.

# The bookkeeping device-kernel library. Most of its kernels are reductions
# over a SCATTERED index set (argmax / min-ratio), hand-launched; the
# elementwise ones (iota / gather / scatter / line step) ride
# wwr.extension.parallel_for, so it links that INTERFACE library (which carries
# wwr.device + the bridge include root) PRIVATE, like calaman.laqps.
calaman_add_gpu_device_library(
  NAME
  calaman.nnls.device
  SOURCES
  nnls.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
)

# nnls() is instantiated once here (IMPLEMENTATION instantiations.cpp + the
# `extern template` in nnls.cppm), so an importer links the symbol and never
# instantiates a body that names the .cu launchers (the calaman.lacpy pattern).
# PUBLIC links are the modules whose types appear in the EXPORTED signatures
# (the two handle types, the status); all the body alone names is PRIVATE.
# nnls.cppm's GMF includes nnls_bridge.h (INCLUDE_DIRS_PRIVATE, this dir), which
# pulls runtime.h; a GMF cannot import, so wwrStream_t arrives by #include and
# the backend define is read via LINK_PRIVATE wwr_backend.
#   wwr.blas            handle, status, WWRBLAS_OP_* / FILL_* / SIDE_*
#   wwr.solver          wwrsolverDnHandle_t, WWRSOLVER_STATUS_SUCCESS
#   wwr.wrappers.blas   gemv, trsv, nrm2
#   wwr.wrappers.solver ormqr, ormqr_bufferSize
#   wwr.runtime_api     wwrMemcpy(Async) / wwrMemsetAsync / wwrStreamSynchronize
#   calaman.common      WorkspaceLayout, carve_workspace
#   calaman.geqp3       geqp3, geqp3_work_size
calaman_add_cxx_module_library(
  NAME
  calaman.nnls
  PRIMARY_INTERFACE
  nnls.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr.blas
  wwr.solver
  LINK_PRIVATE
  wwr.wrappers.blas
  wwr.wrappers.solver
  wwr.runtime_api
  calaman.common
  calaman.geqp3
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::nnls_* kernel
# instantiations are dropped (the module only names them through
# lazily-resolved calls) and a consumer fails to link -- the whole-archive link
# lacpy needs.
target_link_libraries(
  calaman.nnls
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.nnls.device>"
)
