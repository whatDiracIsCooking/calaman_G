# calaman.larnv -- a random vector from LAPACK's stream (?larnv): every
# 64-output chunk in parallel from a jump-ahead seed over the laruv_block
# header, plus one block that replays the tail after a single-precision retry.

# The device-kernel library. laruv::header carries "lapack/laruv/laruv.cuh"
# and wwr.device; calaman.common the src/ root for "common/elem_ops.cuh"
# (make_complex, real_type) and, through it, wrappers/math/math.cuh.
calaman_add_gpu_device_library(
  NAME
  calaman.larnv.device
  SOURCES
  larnv.cu
  LINK_PRIVATE
  calaman::laruv::header
  calaman::common
)

# The module: host C++23 over the launcher declared in larnv_bridge.h, which
# interface.cppm includes in its GMF (hence INCLUDE_DIRS_PRIVATE, and
# wwr_backend PRIVATE for the selected_backend.h behind runtime.h there).
# PUBLIC links are what the exported surface names: wwrStream_t, the complex
# types, usual_fp (calaman.common) and Status (calaman.error_handling).
calaman_add_cxx_module_library(
  NAME
  calaman.larnv
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr::runtime_api
  wwr::complex
  calaman::common
  calaman::error_handling
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::larnv<T>
# instantiations are dropped and a consumer fails to link.
calaman_link_device_archive(calaman.larnv calaman.larnv.device)
