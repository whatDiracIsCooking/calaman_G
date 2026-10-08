# calaman.la_geamv -- y := alpha * |op(A)| * |x| + beta * |y| for a general A,
# with the symbolic-zero nudge: the GPU counterpart of LAPACK's ?la_geamv.

# calaman.la_amv -- la_amv.cuh, the per-y(i) body shared by every ?la_*amv
# kernel (la_geamv, la_gbamv). Not a module: a kernel reaches it by #include,
# as "lapack/la_geamv/la_amv.cuh" off the src/ root calaman.common exports.
calaman_add_interface_library(
  NAME
  calaman.la_amv
  LINK_PUBLIC
  calaman::common
)

# The device-kernel library: one thread per element of y. calaman.common carries
# the src/ root for common/align_up.h and common/enums.h plus, through it,
# wwr.device (runtime.h, complex.h, math.cuh).
calaman_add_gpu_device_library(
  NAME
  calaman.la_geamv.device
  SOURCES
  la_geamv.cu
  LINK_PRIVATE
  calaman::la_amv
)

# The module: host C++23 over the launcher declared in la_geamv_bridge.h, which
# interface.cppm includes in its GMF (hence INCLUDE_DIRS_PRIVATE, and
# wwr::backend PRIVATE for the selected_backend.h behind runtime.h there).
# PUBLIC links are what the exported surface names: wwrStream_t, the complex
# element types, Trans and ComplexToRealType (calaman.common) and Status.
calaman_add_cxx_module_library(
  NAME
  calaman.la_geamv
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

# Force the device archive members in, or the explicit device::la_geamv<T, R>
# instantiations are dropped and a consumer fails to link -- as for lagtm.
calaman_link_device_archive(calaman.la_geamv calaman.la_geamv.device)
