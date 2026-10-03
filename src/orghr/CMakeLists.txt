# calaman.orghr -- generate the orthogonal Q from ?gehrd's Householder
# reflectors, the GPU counterpart of LAPACK's ?orghr (real s/d). A host driver:
# it does NOT reimplement ?orgqr but shuffles the reflectors one column right,
# identity-pads the window's complement, then hands the window block to
# wwr::orgqr. The reflector shuffle is the hand-launched device kernel below.

# The device-kernel library. The kernel is hand-launched (a 2-D grid of 1-D
# blocks over the whole matrix, laset's layout), so it links the pieces directly
# rather than riding wwr.extension.parallel_for: wwr.device carries WarpWraps's
# src include root (runtime.h, which orghr_bridge.h and orghr.cu pull in for
# wwrStream_t and WWR_WARP_SIZE) plus the runtime libs; calaman.common exports
# the src/ root that resolves "common/align_up.h" (idivup, the x-block count).
# The link is PRIVATE, as every device-library link is.
calaman_add_gpu_device_library(
  NAME
  calaman.orghr.device
  SOURCES
  orghr.cu
  LINK_PRIVATE
  wwr.device
  calaman.common
)

# The module itself: ordinary host C++23, no device code. instantiations.cpp is
# the explicit-instantiation implementation unit paired with interface.cppm's
# `extern template` list (the calaman.lacpy pattern).
#
# INCLUDE_DIRS_PRIVATE is for orghr_bridge.h, which interface.cppm includes in
# its global module fragment; LINK_PRIVATE wwr_backend is for the runtime.h that
# header pulls in there (a GMF cannot import, so wwrStream_t arrives by #include
# and selected_backend.h reads WWR_GPU_BACKEND_{CUDA,HIP}).
#
# PUBLIC links are the modules whose types appear in the EXPORTED signatures:
# wwr.solver (the wwrsolverDnHandle_t) and calaman.error_handling (the Status
# return type, re-exported via `export import`, and the src/ root that resolves
# "error_handling/error_macros.h" the GMF pulls in for CLM_TRY). Everything the
# body alone names is PRIVATE.
#   wwr.blas            WWRBLAS_STATUS_INVALID_VALUE / _INTERNAL_ERROR
#   wwr.runtime_api     wwrMemcpyAsync / wwrGetLastError / wwrSuccess
#   wwr.wrappers.solver orgqr, orgqr_bufferSize
#   calaman.common      WorkspaceLayout
calaman_add_cxx_module_library(
  NAME
  calaman.orghr
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr.solver
  calaman.error_handling
  LINK_PRIVATE
  wwr.blas
  wwr.runtime_api
  wwr.wrappers.solver
  calaman.common
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::orghr_prep<T>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the whole-archive link lacpy needs.
target_link_libraries(
  calaman.orghr
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.orghr.device>"
)
