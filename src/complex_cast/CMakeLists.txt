# calaman.complex_cast -- GPU-parallel conversion between real and complex
# floating-point device arrays (set/get the real or imaginary component). A
# shipped src/ module: written once against WarpWraps's `wwr*` names and built
# for either backend.
#
# Not a LAPACK routine -- it is the packing glue a complex solver needs to
# assemble a complex array from two real planes, or split one apart -- so it is
# its own module, like calaman.horner. Structurally the twin of calaman.gebal's
# kernel half: four hand-written per-element functors launched through
# wwr.extension.parallel_for, reading/writing complex components through
# complex.h's neutral accessors (never .x/.y, which hipComplex lacks).

# The device-kernel library. complex_cast.cu launches its four functors through
# wwr.extension.parallel_for, so it links that INTERFACE library PRIVATE, which
# carries parallel_for.cuh plus, transitively, wwr.device: complex.h for the
# neutral complex types and runtime.h (wwrStream_t, pulled in by
# complex_cast_bridge.h), plus the backend runtime libs. It reads/writes complex
# components through elem_ops's real_part/imag_part/make_complex
# (common/elem_ops.cuh), so it links calaman.common too, for the src/ root that
# resolves that header (and, transitively via wwr.device, the complex.h behind
# it).
calaman_add_gpu_device_library(
  NAME
  calaman.complex_cast.device
  SOURCES
  complex_cast.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
  calaman.common
)

# The module itself: ordinary host C++23, no device code. interface.cppm carries
# the four exported wrappers; instantiations.cpp is the explicit-instantiation
# implementation unit paired with its `extern template` list.
#
# LINK_PUBLIC carries every module the exported surface names, so an importer
# sees them: wwr.runtime_api (the wwrStream_t each forwards), wwr.complex
# (the two complex element types) and calaman.common (the complex_fp
# constraint and the ComplexToRealType the real arrays are spelled with).
#
# LINK_PRIVATE wwr_backend is for the runtime.h that
# complex_cast_bridge.h pulls into the global module fragment: a GMF cannot
# `import`, so wwrStream_t arrives by #include, and selected_backend.h behind it
# reads WWR_GPU_BACKEND_{CUDA,HIP}. INCLUDE_DIRS_PRIVATE is for
# complex_cast_bridge.h itself (included from the GMF).
calaman_add_cxx_module_library(
  NAME
  calaman.complex_cast
  PRIMARY_INTERFACE
  interface.cppm
  IMPLEMENTATION
  instantiations.cpp
  LINK_PUBLIC
  wwr.runtime_api
  wwr.complex
  calaman.common
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::*_part launcher
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the same whole-archive link gebal and
# lacpy need. The install-interface spelling waits on the install tier
# (CALAMAN_INSTALL, docs/architecture.md §2); this build-interface link is what
# the tests need meanwhile.
target_link_libraries(
  calaman.complex_cast
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.complex_cast.device>"
)
