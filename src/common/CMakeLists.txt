# calaman.common -- the shared building blocks, as ONE module split into
# partitions, plus one device-only header (elem_ops.cuh) that rides along on the
# same src/ include root:
#
#   common.cppm         primary interface; only `export import`s the partitions
#   :constants          constants.cppm + constants.h -- typed constants
#   :align_up           align_up.cppm + align_up.h -- integer rounding helpers
#   :enums              enums.cppm + enums.h -- the LAPACK selector enums
#   :workspace          workspace.cppm -- pointer carving (WorkspaceLayout)
#                       and the carve-once idiom (SlicesFor, carve_workspace)
#                       for one device scratch buffer; pure host arithmetic
#   :validation         validation.cppm -- argument-checking predicates
#                       (all_nonnull); pure, constexpr, header-less
#
# The :constants, :align_up and :enums partitions have no device code and no
# implementation unit: each is a header-plus-partition pair whose values,
# helpers or enums are `inline constexpr` / `constexpr` / scoped enums in the
# .h, while the .cppm only re-exports them (includes the header in its GMF and
# republishes the names). The .h is a
# GMF-shared header, #included both by other modules and by device .cu TUs as
# "common/<name>.h", so the include root that resolves that spelling must be
# EXPORTED, not PRIVATE (CLAUDE.md, Conventions). These are the first shared
# headers in src/ -- the lacpy bridge is internal, so it kept its root PRIVATE.
# :workspace and :validation have no header -- nothing device-side needs
# them -- so each is just the .cppm.
#
# elem_ops.cuh is NOT a partition: it is a device-only `.cuh` (per-element
# scalar arithmetic over float/double/complex) #included into consumers' .cu
# TUs and never imported, so it has no module face. It lives here only to share
# the src/ include root -- reached root-relative as "common/elem_ops.cuh" --
# having been folded in from the former standalone calaman.elem_ops INTERFACE
# library. Its WarpWraps device dependency is why calaman.common is no longer
# dependency-free (see LINK_PUBLIC below).
#
# INCLUDE_DIRS_PUBLIC is the src/ root, wrapped in $<BUILD_INTERFACE:> so the
# raw build path does not leak into the install interface; the install-interface
# spelling waits on the install tier (CALAMAN_INSTALL, docs/architecture.md §2),
# as it does for lacpy.
#
# LINK_PUBLIC wwr.device: elem_ops.cuh #includes complex.h and
# wrappers/math/math.cuh, whose include roots and backend runtime libs ride on
# wwr.device. PUBLIC so a device-.cu consumer that reaches the header through
# this target resolves them transitively -- the same role the deleted
# calaman.elem_ops target played. This is the one dependency calaman.common
# carries; every module partition remains self-contained.
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
  align_up.cppm
  enums.cppm
  workspace.cppm
  validation.cppm
  INCLUDE_DIRS_PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  LINK_PUBLIC
  wwr.device
  IMPORT_STD
)
