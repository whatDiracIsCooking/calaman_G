# calaman.linalg -- the dense linear-algebra routines (the geqp3 call graph:
# larfg, larf, laqp2, geqp3, laqps), as ONE module split into partitions:
#
#   linalg.cppm   primary interface; only `export import`s the partitions
#
# Backend-neutral by construction, like every src/ module: written once against
# WarpWraps's `wwr*` names, added once here, built for whichever backend
# CALAMAN_GPU_BACKEND selected. The routines compose from the OUTERMOST
# WarpWraps layer that does the job (wwr.wrappers.blas / wwr.wrappers.solver),
# reaching a lower layer only behind a switch that answers for both backends.
#
# Each routine PR adds its partition file to PARTITIONS here and the matching
# `export import` in linalg.cppm, plus any LINK_PUBLIC the new partition's body
# names (the routine bodies are defined in their interface partitions, like
# calaman.diff_norm, so every module a body names must reach an importer --
# hence PUBLIC, not PRIVATE). INCLUDE_DIRS_PUBLIC is the src/ root so a
# partition can include a calaman.common GMF-shared header ("common/<name>.h")
# when it needs one.
#
# Partitions and the LINK_PUBLIC each adds (union below):
#   :larfg  larfg.cppm -- LAPACK's elementary Householder reflector, composed
#           from wwr.wrappers.blas (nrm2, scal) with the scalar arithmetic on
#           the host. Needs wwr.blas (handle/status) and wwr.runtime_api
#           (wwrMemcpy to read alpha / write beta back); uses std::sqrt, hence
#           IMPORT_STD.
#   :larf   larf.cppm -- apply an elementary reflector (?larf) as one gemv plus
#           one ger. Needs wwr.wrappers.blas (gemv, ger), wwr.blas (handle,
#           status, WWRBLAS_OP_* constants) and calaman.common (kZero/kOne
#           scalars, the Side enum).
#   :laqp2  laqp2.cppm -- the level-2 Businger-Golub pivoted-QR panel (?laqp2),
#           a host composition of larfg/larf plus wwr.wrappers.blas (iamax,
#           swap, nrm2); its one device stage, the LAWN 176 partial-norm
#           downdate, is
#           the device library below. The partition's GMF includes
#           laqp2_bridge.h (gpu_stream_bridge.h -> selected_backend.h), so like
#           lacpy it links wwr_backend PRIVATE and reaches the src/ root through
#           the PUBLIC include dir already set for the other partitions.
#   :laqps  laqps.cppm -- the level-3 blocked Businger-Golub pivoted-QR panel
#           (?laqps), a host composition of larfg plus wwr.wrappers.blas (iamax,
#           swap, gemv, gemm, nrm2) that accumulates the auxiliary matrix F and
#           defers the trailing-matrix update to one gemm per block. Its one
#           device stage, the deferred LAWN 176 partial-norm downdate (plus a
#           one-shot flag clear), is the second device library below. Its GMF
#           includes laqps_bridge.h (gpu_stream_bridge.h), so like :laqp2 it
#           reaches wwr_backend PRIVATE and the src/ root through the PUBLIC
#           include dir.
#   :geqp3  geqp3.cppm -- the top-level QR-with-column-pivoting driver (?geqp3),
#           all-free BLOCKED path: seeds the column norms vn1/vn2 with one
#           wwr::nrm2 per column, then runs :laqps blocks (finishing the tail
#           with :laqp2) above a small crossover, else one :laqp2 panel. A pure
#           host composition; needs nothing :laqp2 / :laqps do not already link.

# The downdate device-kernel libraries -- hand-launched through
# wwr.extension.parallel_for from laqp2.cu / laqps.cu, so each links that
# INTERFACE library (which carries wwr.device + the bridge include root)
# PRIVATE, like WarpWraps's own parallel_for consumers.
calaman_add_gpu_device_library(
  NAME
  calaman.linalg.laqp2.device
  SOURCES
  laqp2.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
)

calaman_add_gpu_device_library(
  NAME
  calaman.linalg.laqps.device
  SOURCES
  laqps.cu
  LINK_PRIVATE
  wwr.extension.parallel_for
)

calaman_add_cxx_module_library(
  NAME
  calaman.linalg
  PRIMARY_INTERFACE
  linalg.cppm
  PARTITIONS
  larfg.cppm
  larf.cppm
  laqp2.cppm
  laqps.cppm
  geqp3.cppm
  LINK_PUBLIC
  wwr.wrappers.blas
  wwr.blas
  wwr.runtime_api
  calaman.common
  LINK_PRIVATE
  wwr_backend
  INCLUDE_DIRS_PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  INCLUDE_DIRS_PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  IMPORT_STD
)

# Force the device archive members in, or the explicit
# device::laqp2_downdate<T> / device::laqps_downdate<T> instantiations are
# dropped (the module only names them through lazily-resolved calls) and a
# consumer fails to link -- the same whole-archive link lacpy needs.
target_link_libraries(
  calaman.linalg
  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.linalg.laqp2.device>"
         "$<LINK_LIBRARY:WHOLE_ARCHIVE,calaman.linalg.laqps.device>"
)
