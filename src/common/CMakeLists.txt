# calaman.common.constants -- typed mathematical constants (kZero .. kPi), a
# header-plus-module pair with no device code and no implementation unit: the
# values are `inline constexpr` in constants.h, and this module only re-exports
# them (constants.cppm includes the header in its GMF and republishes them).
#
# INCLUDE_DIRS_PUBLIC is the src/ root, and PUBLIC on purpose. constants.h is a
# GMF-shared header meant to be #included by other modules and by device .cu TUs
# as "common/constants.h", so the include root that resolves that spelling must
# be EXPORTED, not PRIVATE (CLAUDE.md, Conventions). This is the first shared
# header in src/ -- lacpy's bridge is internal, so it kept its root PRIVATE.
#
# Wrapped in $<BUILD_INTERFACE:> so the raw build path does not leak into the
# install interface; the install-interface spelling waits on the install tier
# (CALAMAN_INSTALL, docs/architecture.md §2), as it does for lacpy.
calaman_add_cxx_module_library(
  NAME
  calaman.common.constants
  PRIMARY_INTERFACE
  constants.cppm
  INCLUDE_DIRS_PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
