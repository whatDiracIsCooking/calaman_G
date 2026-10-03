# calaman.expm -- the matrix exponential exp(A) by scaling and squaring with a
# diagonal Pade approximant (Higham's Algorithm 2.3). A shipped src/ module: a
# host composition over wrapped BLAS (gemm/geam) and the wrapped LU solve
# (getrf/getrs), plus four fused device kernels of its own,
# written once against WarpWraps's `wwr*` names and built for either backend.
#
# The matrix exponential is not a LAPACK routine (LAPACK ships no expm), so it
# is its own module rather than a calaman.linalg partition, like calaman.horner.

# The device-kernel library: the four genuinely per-element pieces (the fused
# Pade evaluation, the numerator/denominator split, and the two halves of the
# 1-norm). expm.cu launches its own functors
# through wwr.extension.parallel_for, defers the column sums to
# calaman.reduce_columns's header, and takes its type-generic scalar arithmetic
# from calaman.elem_ops's header, so it links all three PRIVATE: parallel_for
# carries parallel_for.cuh (plus, transitively, wwr.device for complex.h);
# reduce_columns, the src/ include root for reduce_columns.cuh plus its own
# wwr.device dep; elem_ops, the src/ root for elem_ops/elem_ops.cuh (which pulls
# complex.h and wrappers/math/math.cuh via wwr.device). The link is PRIVATE, as
# every device-library link is.
calaman_add_gpu_device_library(
  NAME
  calaman.expm.device
  SOURCES
  expm.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
  calaman.reduce_columns
  calaman.elem_ops
)

# The module itself: ordinary host C++23, no device code. instantiations.cpp is
# the explicit-instantiation implementation unit paired with interface.cppm's
# `extern template` list.
#
# LINK_PUBLIC is the union of what the exported surface names: the BLAS and
# solver handle/status/enum layers (wwr.blas, wwr.solver), the wwrStream_t and
# memcpy/sync (wwr.runtime_api), the complex element types and their host
# constructors (wwr.complex), the usual_fp/ComplexToRealType traits
# (wwr.wrappers.common), the type-safe BLAS and solver wrappers
# (wwr.wrappers.blas, wwr.wrappers.solver), the ScopedPointerMode guard the
# pade/expm bodies force host mode with (wwr.extension.blas) and the
# WorkspaceLayout/align_up (calaman.common). calaman.error_handling is
# additionally re-exported (export import) because expm/pade RETURN
# calaman::Status -- and it carries PointerModeStatus, the guard's error policy.
#   calaman.error_handling  Status, the cross-domain return type
# LINK_PRIVATE wwr_backend is for the runtime.h that expm_bridge.h pulls into
# the global module fragment: a GMF cannot `import`, so wwrStream_t arrives by
# #include and selected_backend.h behind it reads WWR_GPU_BACKEND_*.
# INCLUDE_DIRS_PRIVATE is the source dir for expm_bridge.h itself.
calaman_add_cxx_module_library(
  NAME
  calaman.expm
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  detail.cppm
  plan.cppm
  norm1.cppm
  buffer_size.cppm
  pade.cppm
  expm.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr.blas
  wwr.solver
  wwr.runtime_api
  wwr.complex
  wwr.wrappers.common
  wwr.wrappers.blas
  wwr.wrappers.solver
  wwr.extension.blas
  calaman.common
  calaman.error_handling
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::* launcher
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the same whole-archive link lacpy,
# horner and columnwise_ell1 need. The install-interface spelling waits on the
# install tier (CALAMAN_INSTALL, docs/architecture.md §2); this build-interface
# link is what the tests need meanwhile.
target_link_libraries(
  calaman.expm
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.expm.device>"
)
