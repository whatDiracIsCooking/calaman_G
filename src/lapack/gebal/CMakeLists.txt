# calaman.gebal -- matrix balancing for the nonsymmetric eigenproblem, the GPU
# counterpart of LAPACK's ?gebal. A shipped src/ module: written once against
# WarpWraps's `wwr*` names and built for either backend.
#
# Unlike the calaman.linalg routines, gebal is NOT a BLAS composition: both
# stages are hand-written kernels. The permutation search is a full-grid scan;
# the scaling sweep is a single cooperating block because it is Gauss-Seidel in
# the index. The host driver -- which branches on device data each step -- lives
# in the module (interface.cppm), reaching the kernels through the launchers in
# gebal_bridge.h, the same module/.cu split lacpy uses.
#
# This is the project's first complex-capable routine: balancing only measures
# magnitudes (CABS1) and applies real scales, so complex is a genuine path, not
# the material-difference case larfg/laqp2 defer. gebal.cu therefore pulls in
# WarpWraps's neutral complex (complex.h) via wwr.device.

# The device-kernel library. gebal.cu seeds scale[] through
# wwr.extension.parallel_for and launches the rest of its kernels by hand (raw
# <<<>>>), so it links that INTERFACE library PRIVATE -- which carries
# parallel_for.cuh plus, transitively, wwr.device: complex.h for the neutral
# complex types and runtime.h (wwrStream_t, pulled in by gebal_bridge.h), plus
# the backend runtime libs. It also takes its per-element scale/abs1/is_zero
# from calaman.elem_ops's header, so it links that INTERFACE library too, for
# the src/ root that resolves "elem_ops/elem_ops.cuh" (math.cuh via wwr.device).
calaman_add_gpu_device_library(
  NAME
  calaman.gebal.device
  SOURCES
  gebal.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
  calaman.elem_ops
)

# The module itself: ordinary host C++23, no device code. interface.cppm carries
# the exported API and the host driver; instantiations.cpp is the explicit-
# instantiation implementation unit paired with its `extern template` list.
#
# LINK_PUBLIC carries every module the exported surface names, so an importer
# sees them: wwr.runtime_api (the stream/error types and the memcpy, memset and
# sync the driver issues), wwr.complex (the two complex element types),
# wwr.wrappers.common (the usual_fp constraint and ComplexToRealType the scale
# array's type is spelled with) and calaman.error_handling (gebal re-exports it
# for Status, its return type).
#
# LINK_PRIVATE wwr_backend is for the runtime.h that gebal_bridge.h pulls into
# the global module fragment: a GMF cannot `import`, so wwrStream_t arrives by
# #include, and selected_backend.h behind it reads WWR_GPU_BACKEND_{CUDA,HIP}
# with no device pass to read __CUDACC__/__HIP__.
# INCLUDE_DIRS_PRIVATE is for gebal_bridge.h itself (included from the GMF).
calaman_add_cxx_module_library(
  NAME
  calaman.gebal
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr.runtime_api
  wwr.complex
  wwr.wrappers.common
  calaman.error_handling #  Status, the cross-domain return type
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::gebal_* launcher
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the same whole-archive link lacpy and
# linalg need. The install-interface spelling waits on the install tier
# (CALAMAN_INSTALL, docs/architecture.md §2); this build-interface link is what
# the tests need meanwhile.
target_link_libraries(
  calaman.gebal
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.gebal.device>"
)
