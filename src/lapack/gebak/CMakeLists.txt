# calaman.gebak -- backward transformation of eigenvectors, the GPU counterpart
# of LAPACK's ?gebak and the companion of calaman.gebal. A shipped src/ module:
# written once against WarpWraps's `wwr*` names, built for either backend.
#
# Like gebal, gebak is NOT a BLAS composition: both stages are hand-written
# kernels. Scaling is a full-grid pass; the permutation is a single cooperating
# block because it is order-dependent in the index. The host driver -- which,
# unlike gebal's, reads NOTHING back from the device -- lives in the module
# (interface.cppm), reaching the kernels through the launchers in
# gebak_bridge.h, the same module/.cu split lacpy and gebal use.
#
# Complex is a genuine path (the back-transform only scales by real factors and
# moves rows), so gebak.cu pulls in WarpWraps's neutral complex (complex.h).

# The device-kernel library. gebak.cu launches its kernels by hand (raw <<<>>>)
# and takes its per-element scale from calaman.common's elem_ops.cuh header, so
# it links calaman.common PRIVATE -- which carries the src/ root that resolves
# "common/elem_ops.cuh" plus, transitively via wwr.device, complex.h (the
# neutral complex types) and runtime.h (wwrStream_t, pulled in by
# gebak_bridge.h) and the backend runtime libs. No parallel_for here: unlike
# gebal there is no fill helper, so calaman.common is the one device dep.
calaman_add_gpu_device_library(
  NAME
  calaman.gebak.device
  SOURCES
  gebak.cu
  LINK_PRIVATE
  calaman::common
)

# The module itself: ordinary host C++23, no device code. interface.cppm carries
# the exported API and the host driver; instantiations.cpp is the explicit-
# instantiation implementation unit paired with its `extern template` list.
#
# LINK_PUBLIC carries every module the exported surface names, so an importer
# sees them: wwr.runtime_api (the stream/error types and wwrGetLastError the
# driver checks), wwr.complex (the two complex element types),
# calaman.common (usual_fp and ComplexToRealType, the scale array's type)
# and calaman.error_handling (gebak re-exports it for Status, its return type).
#
# LINK_PRIVATE wwr_backend is for the runtime.h that gebak_bridge.h pulls into
# the global module fragment: a GMF cannot `import`, so wwrStream_t arrives by
# #include and selected_backend.h behind it reads WWR_GPU_BACKEND_{CUDA,HIP}
# with no device pass. INCLUDE_DIRS_PRIVATE is for gebak_bridge.h in the GMF.
calaman_add_cxx_module_library(
  NAME
  calaman.gebak
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr::runtime_api
  wwr::complex
  calaman::common
  calaman::error_handling #  Status, the cross-domain return type
  LINK_PRIVATE
  wwr::backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::gebak_* launcher
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the same whole-archive link lacpy and
# gebal need. The install-interface spelling waits on the install tier
# (CALAMAN_INSTALL, docs/architecture.md §2); this build-interface link is what
# the tests need meanwhile.
calaman_link_device_archive(calaman.gebak calaman.gebak.device)
