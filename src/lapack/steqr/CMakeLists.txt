# calaman.steqr -- all eigenvalues and, optionally, eigenvectors of a symmetric
# tridiagonal by the implicit QL/QR method, LAPACK's ?steqr. One single-block
# kernel: thread 0 runs the chase, the block applies its rotations to Z.

# The device-kernel library. The kernel #includes the per-thread helpers of
# the modules it calls -- lanst_max_abs, lartg_scalar, lasr_block and the
# lae2/laev2/lapy2 ports -- through their INTERFACE header targets (each the
# src/ root); calaman.common carries common/enums.h plus wwr.device.
calaman_add_gpu_device_library(
  NAME
  calaman.steqr.device
  SOURCES
  steqr.cu
  LINK_PRIVATE
  calaman::common
  calaman::lanst::header
  calaman::lartg::header
  calaman::lasr::header
  calaman::sym2x2
)

# The module: host C++23 over the launcher declared in steqr_bridge.h, which
# interface.cppm includes in its GMF (hence INCLUDE_DIRS_PRIVATE, and
# wwr_backend PRIVATE for the selected_backend.h behind runtime.h there).
# PUBLIC links are what the exported surface names: wwrStream_t, CompZ,
# ComplexToRealType and the workspace carve (calaman.common), wwrFloatComplex/
# wwrDoubleComplex (wwr.complex) and Status (calaman.error_handling).
calaman_add_cxx_module_library(
  NAME
  calaman.steqr
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

# Force the device archive members in, or the explicit device::steqr<T>
# instantiations are dropped and a consumer fails to link -- as for lasr.
calaman_link_device_archive(calaman.steqr calaman.steqr.device)
