# calaman.linalg -- the dense linear-algebra routines (the geqp3 call graph:
# larfg, larf, laqp2, geqp3, laqps), as ONE module split into partitions:
#
#   linalg.cppm   primary interface; only `export import`s the partitions
#
# Backend-neutral by construction, like every src/ module: written once against
# WarpWraps's `wwr*` names, added once here, built for whichever backend
# CALAMAN_GPU_BACKEND selected. The routines compose from the OUTERMOST WarpWraps
# layer that does the job (wwr.wrappers.blas / wwr.wrappers.solver), reaching a
# lower layer only behind a switch that answers for both backends.
#
# This is the assembly point and starts with no PARTITIONS: each routine PR adds
# its partition file here and the matching `export import` in linalg.cppm, plus
# any LINK_PUBLIC the new partition's body names (the routine bodies are defined
# in their interface partitions, like calaman.diff_norm, so every module a body
# names must reach an importer -- hence PUBLIC, not PRIVATE). INCLUDE_DIRS_PUBLIC
# is the src/ root so a partition can include a calaman.common GMF-shared header
# ("common/<name>.h") when it needs one.
calaman_add_cxx_module_library(
  NAME
  calaman.linalg
  PRIMARY_INTERFACE
  linalg.cppm
  INCLUDE_DIRS_PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
