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
# hence PUBLIC, not PRIVATE). INCLUDE_DIRS_PUBLIC is the src/ root so a partition
# can include a calaman.common GMF-shared header ("common/<name>.h") when needed.
#
# :larfg (larfg.cppm) is the first partition -- LAPACK's elementary Householder
# reflector, composed from wwr.wrappers.blas (nrm2, scal) with the scalar
# arithmetic on the host. Its links:
#   wwr.wrappers.blas   nrm2, scal
#   wwr.blas            wwrblasHandle_t / wwrblasStatus_t / WWRBLAS_STATUS_*
#   wwr.runtime_api     wwrMemcpy + wwrSuccess (read alpha, write beta back)
# IMPORT_STD: larfg uses std::sqrt.
calaman_add_cxx_module_library(
  NAME
  calaman.linalg
  PRIMARY_INTERFACE
  linalg.cppm
  PARTITIONS
  larfg.cppm
  LINK_PUBLIC
  wwr.wrappers.blas
  wwr.blas
  wwr.runtime_api
  INCLUDE_DIRS_PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  IMPORT_STD
)
