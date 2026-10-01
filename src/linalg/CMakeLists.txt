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
calaman_add_cxx_module_library(
  NAME
  calaman.linalg
  PRIMARY_INTERFACE
  linalg.cppm
  PARTITIONS
  larfg.cppm
  larf.cppm
  LINK_PUBLIC
  wwr.wrappers.blas
  wwr.blas
  wwr.runtime_api
  calaman.common
  INCLUDE_DIRS_PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  IMPORT_STD
)
