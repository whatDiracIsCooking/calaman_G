# calaman.feast -- the FEAST eigensolver for real symmetric matrices (Polizzi,
# Phys. Rev. B 79 (2009) 115112, arXiv:0901.2665): the eigenpairs with
# eigenvalues in [Emin, Emax], by subspace iteration with a rational filter
# built from Ne shifted solves along a contour around the interval. A shipped
# src/ module -- a host driver over wrapped batched BLAS, symm/gemm, the wrapped
# eigensolver and calaman.orthogonalize, plus its own fused kernels -- written
# once against WarpWraps's `wwr*` names and built for either backend.
#
# Not a LAPACK routine (LAPACK ships no FEAST), so it is its own module, like
# calaman.expm. One module with six partitions: the inter-partition edges are
# genuine internal pieces of one driver, not reusable routines, so they are
# partitions rather than separate modules (unlike the geqp3 call graph).

# The device-kernel library: the pieces no BLAS/eigensolver call expresses --
# building the shifted resolvents, moving between the real subspace and the Ne
# complex right-hand sides, the symmetric 1-norm, and the per-iteration
# selection and residuals. feast.cu launches raw <<<>>> kernels (no
# parallel_for), so it links wwr.device directly PRIVATE -- the minimal carrier
# of complex.h (neutral complex types/accessors), runtime.h (wwrStream_t via
# feast_bridge.h, plus the WWR_WARP_SIZE block-size define) and the backend
# runtime lib. The link is PRIVATE, as every device-library link is.
calaman_add_gpu_device_library(
  NAME
  calaman.feast.device
  SOURCES
  feast.cu
  LINK_PRIVATE
  wwr.device
)

# The module itself: ordinary host C++23, no device code. interface.cppm
# re-exports the six partitions; the host driver and the workspace layout live
# in them. The template surface is header-only (like the geqp3 family),
# instantiated on use, so there is no instantiations.cpp.
#
# LINK_PUBLIC is the union of what the exported surface names: the runtime
# stream/memcpy/sync and wwrGetLastError (wwr.runtime_api), the BLAS and solver
# handle/status/enum layers (wwr.blas, wwr.solver), the usual_fp/real_fp and
# RealToComplexType traits (wwr.wrappers.common), the type-safe BLAS and
# eigensolver wrappers (wwr.wrappers.blas, wwr.wrappers.solver), the typed
# constants kOne/kZero/kPi and align_up (calaman.common) and the QR
# orthonormalizer (calaman.orthogonalize). wwr.complex is not listed: no
# partition names the vendor complex type, and RealToComplexType's target is
# reachable transitively through wwr.wrappers.common.
#
# LINK_PRIVATE wwr_backend is for the runtime.h that feast_bridge.h pulls into
# each partition's global module fragment: a GMF cannot `import`, so wwrStream_t
# arrives by #include and selected_backend.h behind it reads WWR_GPU_BACKEND_*.
# INCLUDE_DIRS_PRIVATE is the source dir for feast_bridge.h itself.
calaman_add_cxx_module_library(
  NAME
  calaman.feast
  PRIMARY_INTERFACE
  interface.cppm
  PARTITIONS
  feast_quadrature.cppm
  compute_quadrature.cppm
  buffer_size.cppm
  contour_filter.cppm
  rayleigh_ritz.cppm
  feast_solver.cppm
  LINK_PUBLIC
  wwr.runtime_api
  wwr.blas
  wwr.solver
  wwr.wrappers.common
  wwr.wrappers.blas
  wwr.wrappers.solver
  calaman.common
  calaman.orthogonalize
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit device::feast_* launcher
# instantiations are dropped (the module only names them through lazily-resolved
# calls) and a consumer fails to link -- the same whole-archive link lacpy,
# gebal and expm need. The install-interface spelling waits on the install tier
# (CALAMAN_INSTALL, docs/architecture.md §2); this build-interface link is what
# the tests need meanwhile.
target_link_libraries(
  calaman.feast
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.feast.device>"
)
