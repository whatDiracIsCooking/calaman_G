# calaman.common -- the shared, dependency-free building blocks, as ONE module
# split into partitions:
#
#   common.cppm  primary interface; only `export import`s the partitions
#   :constants   constants.cppm + constants.h -- typed constants (kZero .. kPi)
#   :align_up    align_up.cppm + align_up.h -- integer rounding helpers
#
# Each partition is a header-plus-partition pair with no device code and no
# implementation unit: the values/helpers are `inline constexpr` / `constexpr`
# in the .h, and the .cppm only re-exports them (includes the header in its GMF
# and republishes the names). The .h is a GMF-shared header, #included both
# by other modules and by device .cu TUs as "common/<name>.h", so the
# include root that resolves that spelling must be EXPORTED, not PRIVATE
# (CLAUDE.md, Conventions). These are the first shared headers in src/ -- the
# lacpy bridge is internal, so it kept its root PRIVATE.
#
# INCLUDE_DIRS_PUBLIC is the src/ root, wrapped in $<BUILD_INTERFACE:> so the
# raw build path does not leak into the install interface; the install-interface
# spelling waits on the install tier (CALAMAN_INSTALL, docs/architecture.md §2),
# as it does for lacpy.
calaman_add_cxx_module_library(
  NAME
  calaman.common
  PRIMARY_INTERFACE
  common.cppm
  PARTITIONS
  constants.cppm
  align_up.cppm
  INCLUDE_DIRS_PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
