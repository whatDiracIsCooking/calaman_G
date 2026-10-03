# calaman.lascl -- multiply a column-major matrix by cto/cfrom without
# over/underflow on the device, the GPU counterpart of LAPACK's ?lascl for a
# full matrix (TYPE='G'). A shipped src/ module: written once against
# WarpWraps's `wwr*` names, it builds for either backend.

# The device-kernel library. The kernel is hand-launched (a 2-D grid of 1-D
# blocks), so it links the pieces directly rather than riding
# wwr.extension.parallel_for: wwr.device carries WarpWraps's src include root
# (runtime.h, which lascl_bridge.h and lascl.cu pull in for wwrStream_t and
# WWR_WARP_SIZE) plus the runtime libs; calaman.common exports the src/ root
# that resolves "common/align_up.h" (idivup, for the x-direction block count).
# The link is PRIVATE, as every device-library link is.
calaman_add_gpu_device_library(
  NAME
  calaman.lascl.device
  SOURCES
  lascl.cu
  LINK_PRIVATE
  wwr.device
  calaman.common
)

# The module itself: ordinary host C++23, no device code. instantiations.cpp is
# the explicit-instantiation implementation unit paired with interface.cppm's
# `extern template` list; the guarded multiplier loop is host scalar arithmetic
# in the interface, so IMPORT_STD supplies the <limits>/<cmath> names it uses.
#
# INCLUDE_DIRS_PRIVATE is for lascl_bridge.h, which interface.cppm includes in
# its global module fragment. LINK_PRIVATE wwr_backend is for the runtime.h that
# header pulls in there: a GMF cannot `import`, so wwrStream_t arrives by
# #include, and selected_backend.h behind it reads WWR_GPU_BACKEND_{CUDA,HIP}
# with no device pass to read __CUDACC__/__HIP__.
#
# calaman.error_handling is PUBLIC: the interface re-exports it (export import)
# for the Status return type, and its src/ root resolves the
# "error_handling/error_macros.h" interface.cppm pulls into the GMF for CLM_TRY.
calaman_add_cxx_module_library(
  NAME
  calaman.lascl
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr.runtime_api
  calaman.error_handling
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::lascl<T>
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link. The install-interface spelling waits on
# the install tier (CALAMAN_INSTALL, docs/architecture.md §2); this build-
# interface link is what the tests need meanwhile.
target_link_libraries(
  calaman.lascl
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.lascl.device>"
)
