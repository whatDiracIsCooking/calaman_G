# calaman.cg_unitary -- Riemannian conjugate gradient on the unitary group
# U(n) (Abrudan, Eriksson & Koivunen, Signal Processing 89 (2009) 1704-1714).
# A host composition over wrapped BLAS (gemm/gemv/geam/dot/dotc/dgmm/axpy/
# scal/nrm2), calaman.expm for the geodesic step, and calaman.orthogonalize
# for the optional reunitarization, plus its own single-block device kernels:
# the scalar stages of the two geodesic line searches no BLAS call expresses.
#
# The matrix exponential is not a LAPACK routine, and neither is this -- so it
# is its own module, like calaman.expm, not a calaman.linalg partition.

# The line-search device-kernel library. These are reductions and bracketed
# root finds over a handful of samples (the Vandermonde solve, the Hann DFT,
# the two trig/polynomial bracketings, the step selection), all hand-launched
# in single blocks. It names complex.h's device builders and runtime.h's
# launch syntax, both carried (transitively, via wwr.device) by
# wwr.extension.parallel_for -- linked PRIVATE for the include roots, as every
# device-library link is.
calaman_add_gpu_device_library(
  NAME
  calaman.cg_unitary.device
  SOURCES
  cg_unitary.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
)

# The module: ordinary host C++23. Its public functions are templates over the
# element type AND a user cost functor, so -- unlike expm/nnls -- they cannot
# be pre-instantiated here; a consumer instantiates them and links the concrete
# device-kernel instantiations out of the archive below. Every module whose
# templates the consumer's instantiation reaches is therefore PUBLIC.
#   wwr.blas / wwr.solver / wwr.runtime_api  handle, status and stream types
#   wwr.complex                              the complex element types
#   wwr.wrappers.common                      usual_fp / ComplexToRealType / ...
#   wwr.wrappers.blas                        gemm/gemv/geam/dot/dotc/dgmm/...
#   calaman.common                           kOne/kZero/..., WorkspaceBuilder
#   calaman.error_handling                   Status, the cross-domain return
#   calaman.expm                             expm, expm_bufferSize, matrix_norm1
#   calaman.orthogonalize                    the optional reunitarization
# LINK_PRIVATE wwr_backend is for the runtime.h that cg_unitary_bridge.h pulls
# into the global module fragment (a GMF cannot import); INCLUDE_DIRS_PRIVATE
# is this dir, for the bridge header itself.
calaman_add_cxx_module_library(
  NAME
  calaman.cg_unitary
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  cost_function.cppm
  brockett.cppm
  buffer_size.cppm
  geodesic_search.cppm
  cg_solver.cppm
  LINK_PUBLIC
  wwr.blas
  wwr.solver
  wwr.runtime_api
  wwr.complex
  wwr.wrappers.common
  wwr.wrappers.blas
  calaman.common
  calaman.error_handling
  calaman.expm
  calaman.orthogonalize
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::cg_* kernel
# instantiations are dropped (the module only names them through
# lazily-resolved calls) and a consumer fails to link -- the whole-archive link
# lacpy/expm/nnls need.
target_link_libraries(
  calaman.cg_unitary
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.cg_unitary.device>"
)
