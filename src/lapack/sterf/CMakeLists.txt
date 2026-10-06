# calaman.sterf -- eigenvalues of a symmetric tridiagonal (d, e), LAPACK's
# ?sterf, by the root-free QL/QR iteration in one single-thread kernel (the
# calaman.lahqr shape), plus the in-thread sterf_serial header (sterf.cuh) that
# ?syev's kernels can call.

# The header's own target: the src/ root that resolves
# "lapack/sterf/sterf.cuh", plus the device helpers it includes -- ?lae2/?lapy2
# (sym2x2), ?lanst('M') (lanst::header) and the closing ?lasrt (lasrt::header).
add_library(calaman.sterf.header INTERFACE)
add_library(calaman::sterf::header ALIAS calaman.sterf.header)
target_include_directories(
  calaman.sterf.header
  INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:${CALAMAN_INSTALL_INCLUDEDIR}>
)
target_link_libraries(
  calaman.sterf.header INTERFACE calaman::sym2x2 calaman::lanst::header
                                 calaman::lasrt::header
)

# The device-kernel library: one hand-launched single-thread kernel. sym2x2
# (through sterf::header) carries wwr.device -- runtime.h and the runtime libs.
calaman_add_gpu_device_library(
  NAME
  calaman.sterf.device
  SOURCES
  sterf.cu
  LINK_PRIVATE
  calaman::sterf::header
)

# The module: host C++23 over the launcher declared in sterf_bridge.h, which
# interface.cppm includes in its GMF (hence INCLUDE_DIRS_PRIVATE, and
# wwr_backend PRIVATE for the selected_backend.h behind runtime.h there).
# PUBLIC links are what the exported surface names: wwrStream_t, real_fp
# (calaman.common) and Status (calaman.error_handling).
calaman_add_cxx_module_library(
  NAME
  calaman.sterf
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

# Force the device archive members in, or the explicit device::sterf<T>
# instantiations are dropped and a consumer fails to link -- as for lahqr.
calaman_link_device_archive(calaman.sterf calaman.sterf.device)
