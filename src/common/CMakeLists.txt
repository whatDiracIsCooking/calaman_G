# calaman.common -- the shared building blocks, as ONE module split into
# partitions, plus two device-only headers (elem_ops.cuh, block_reduce.cuh)
# that ride along on the same src/ include root:
#
#   common.cppm         primary interface; only `export import`s the partitions
#   :constants          constants.cppm + constants.h -- typed constants
#   :fp_types           fp_types.cppm + fp_types.h -- the fp concepts and
#                       real/complex type maps, re-published under `calaman::`
#                       from WarpWraps' wrappers/common/fp_types.h
#   :align_up           align_up.cppm + align_up.h -- integer rounding helpers
#   :enums              enums.cppm + enums.h -- the LAPACK selector enums
#   :workspace          workspace.cppm -- pointer carving (WorkspaceLayout)
#                       and the carve-once idiom (SlicesFor, carve_workspace)
#                       for one device scratch buffer; pure host arithmetic
#   :validation         validation.cppm -- argument-checking predicates
#                       (all_nonnull); pure, constexpr, header-less
#
# The :constants, :fp_types, :align_up and :enums partitions have no device
# code and no implementation unit: each is a header-plus-partition pair whose
# values, helpers, enums or (for :fp_types) re-exported concepts and type maps
# are `inline constexpr` / `constexpr` / scoped enums / `using`-declarations in
# the .h, while the .cppm only re-exports them (includes the header in its GMF
# and republishes the names). The .h is a
# GMF-shared header, #included both by other modules and by device .cu TUs as
# "common/<name>.h", so the include root that resolves that spelling must be
# EXPORTED, not PRIVATE (CLAUDE.md, Conventions). These are the first shared
# headers in src/ -- the lacpy bridge is internal, so it kept its root PRIVATE.
# :workspace and :validation have no header -- nothing device-side needs
# them -- so each is just the .cppm.
#
# :fp_types differs from the other header pairs in one way: it defines nothing
# of its own, only re-publishing WarpWraps' fp concepts (real_fp, complex_fp,
# usual_fp) and real/complex type maps (ComplexToRealType, RealToComplexType)
# under `calaman::`. Its fp_types.h #includes WarpWraps'
# wrappers/common/fp_types.h (reached via the src/ root wwr.device already
# carries, below), whose complex.h/fp16.h/bf16.h need the backend-selection
# define -- hence the LINK_PRIVATE wwr_backend, mirroring WarpWraps' own
# wwr.wrappers.common.
#
# elem_ops.cuh and block_reduce.cuh are NOT partitions: each is a device-only
# `.cuh` (per-element scalar arithmetic; the block-wide tree reduction)
# #included into consumers' .cu TUs and never imported, so neither has a module
# face. They live here only to share the src/ include root -- reached
# root-relative as "common/<name>.cuh" -- elem_ops.cuh having been folded in
# from the former standalone calaman.elem_ops INTERFACE library. Its WarpWraps
# device dependency is why calaman.common is no longer dependency-free (see
# LINK_PUBLIC below); block_reduce.cuh needs only device_guard.h off the same
# wwr.device root.
#
# INCLUDE_DIRS_PUBLIC is the src/ root, wrapped in $<BUILD_INTERFACE:> so the
# raw build path does not leak into the install interface; the install-interface
# spelling waits on the install tier (CALAMAN_INSTALL, docs/architecture.md §2),
# as it does for lacpy.
#
# LINK_PUBLIC wwr.device: elem_ops.cuh #includes complex.h and
# wrappers/math/math.cuh, and fp_types.h #includes wrappers/common/fp_types.h,
# whose include roots and backend runtime libs ride on wwr.device. PUBLIC so a
# device-.cu consumer that reaches a header through this target resolves them
# transitively -- the same role the deleted calaman.elem_ops target played.
#
# LINK_PRIVATE wwr_backend: the fp_types.cppm partition's GMF #includes
# fp_types.h -> WarpWraps' wrappers/common/fp_types.h ->
# complex.h/fp16.h/bf16.h, which read the WWR_GPU_BACKEND_* backend-selection
# define. PRIVATE because only calaman.common's own partition compiles need it
# (importers see the finished module; a device .cu gets the define from its
# compiler pass). This mirrors WarpWraps' own wwr.wrappers.common. These two
# are the dependencies calaman.common carries; the non-fp_types partitions
# remain self-contained.
#
# IMPORT_STD: the :workspace partition uses std::size_t / std::max via
# `import std;`.
calaman_add_cxx_module_library(
  NAME
  calaman.common
  PRIMARY_INTERFACE
  common.cppm
  PARTITIONS
  constants.cppm
  fp_types.cppm
  align_up.cppm
  enums.cppm
  workspace.cppm
  validation.cppm
  INCLUDE_DIRS_PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  LINK_PUBLIC
  wwr.device
  LINK_PRIVATE
  wwr_backend
  IMPORT_STD
)
