# calaman.inv_sqrt -- the symmetric (Loewdin) inverse square root X = S^{-1/2}
# of a real symmetric PSD matrix: syevd, an inverse_sqrt device kernel (flooring
# near-null modes), then dgmm + gemm to recombine U diag(lambda^{-1/2}) U^T. A
# shipped src/ module, written once against WarpWraps's `wwr*` names and built
# for either backend.

# The device-kernel library: the one per-element step, the flooring rsqrt of the
# eigenvalue vector, launched through wwr.extension.parallel_for. That one link
# carries parallel_for.cuh and wrappers/math/math.cuh (wwr::sqrt) plus runtime.h
# (wwrStream_t) via wwr.device. The link is PRIVATE, as every device-library
# link is.
calaman_add_gpu_device_library(
  NAME
  calaman.inv_sqrt.device
  SOURCES
  inv_sqrt.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
)

# The module itself: ordinary host C++23 driving the wrapped syevd/dgmm/gemm.
# instantiations.cpp is the explicit-instantiation implementation unit paired
# with interface.cppm's `extern template` list.
#
# LINK_PUBLIC is the union of what the exported surface names: the BLAS and
# solver handle/enum layers (wwr.blas, wwr.solver), the wwrStream_t + memcpy
# (wwr.runtime_api), the real_fp concept (wwr.wrappers.common), the type-safe
# BLAS and solver wrappers (wwr.wrappers.blas, wwr.wrappers.solver) and the
# kOne/kZero constants (calaman.common). calaman.error_handling is re-exported
# (export import) for the Status return type, and its src/ root resolves the
# "error_handling/error_macros.h" the GMF pulls in for CLM_TRY.
#
# LINK_PRIVATE wwr_backend is for the runtime.h that inv_sqrt_bridge.h pulls
# into the global module fragment: a GMF cannot `import`, so wwrStream_t
# arrives by #include and selected_backend.h behind it reads WWR_GPU_BACKEND_*.
# INCLUDE_DIRS_PRIVATE is the source dir for inv_sqrt_bridge.h itself.
calaman_add_cxx_module_library(
  NAME
  calaman.inv_sqrt
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr.blas
  wwr.solver
  wwr.runtime_api
  wwr.wrappers.common
  wwr.wrappers.blas
  wwr.wrappers.solver
  calaman.common
  calaman.error_handling
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::inverse_sqrt<T>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link.
target_link_libraries(
  calaman.inv_sqrt
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.inv_sqrt.device>"
)
