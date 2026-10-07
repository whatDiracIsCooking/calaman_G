# calaman.lanst -- the norm of a symmetric tridiagonal (d, e), the GPU
# counterpart of LAPACK's ?lanst, plus the per-thread lanst_max_abs header
# (lanst.h) that ?sterf/?steqr kernels call on each unreduced block.

# The header's own target: an INTERFACE library carrying only the src/ root, so
# a downstream .cu reaches "lapack/lanst/lanst.h" (and the
# "common/host_device.h" it includes) root-relative by linking
# calaman::lanst::header -- reduce_columns's hand-written shape, since
# calaman_add_interface_library would root the path at this directory. It needs
# no other link: the body is compare-and-negate only.
add_library(calaman.lanst.header INTERFACE)
add_library(calaman::lanst::header ALIAS calaman.lanst.header)
target_include_directories(
  calaman.lanst.header
  INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:${CALAMAN_INSTALL_INCLUDEDIR}>
)

# The device-kernel library: one single-block kernel, also calaman.lanht's (a
# complex off-diagonal). calaman.common carries the src/ root for
# common/block_reduce.cuh, common/elem_ops.cuh and common/enums.h plus, through
# it, wwr.device (runtime.h, cooperative_groups.h, complex.h, the runtime libs).
calaman_add_gpu_device_library(
  NAME
  calaman.lanst.device
  SOURCES
  lanst.cu
  LINK_PRIVATE
  calaman::common
  calaman::lanst::header
)

# The module: host C++23 over the launcher declared in lanst_bridge.h, which
# interface.cppm includes in its GMF (hence INCLUDE_DIRS_PRIVATE, and
# wwr_backend PRIVATE for the selected_backend.h behind runtime.h there).
# PUBLIC links are what the exported surface names: wwrStream_t, MatrixNorm
# (re-exported from calaman.common) and Status (calaman.error_handling).
calaman_add_cxx_module_library(
  NAME
  calaman.lanst
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr::runtime_api
  calaman::common
  calaman::error_handling
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::lanst<T>
# instantiations are dropped and a consumer fails to link -- as for lange.
calaman_link_device_archive(calaman.lanst calaman.lanst.device)
