# calaman_install.cmake — install rules, the export set, and the package config.
#
# A C++23 module package ships module interface SOURCES (.cppm), not BMIs (a BMI
# is not portable), and the consumer's own build recompiles them. Three rules
# follow, each enforced below and explained in full in cmake/README.md, "Install
# and the CMake package":
#
#   1. Every compile requirement of a .cppm must reach the consumer — a PRIVATE
#      requirement is not exported, so it becomes a broken install.
#   2. Module sources need per-target destinations (several modules end up
#      rooted at a file of the same name, and one shared dir would collide).
#   3. A header a module unit #includes is installed next to those sources.
#
# The consumer-facing half of the contract (same compiler, standard library and
# backend) is stated and enforced in calamanConfig.cmake.in.

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Where the pieces of the installed package live, all relative to the install
# prefix. `modules` holds module interface SOURCES, not BMIs (see above); it is
# under include/ because that is where a "things the consumer's compiler reads"
# directory belongs, not because anything #includes from it.
set(CALAMAN_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/calaman")
set(CALAMAN_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/calaman")
set(CALAMAN_INSTALL_MODULEDIR "${CALAMAN_INSTALL_INCLUDEDIR}/modules")

# ---------------------------------------------------------------------------
# Internal helpers (underscore-prefixed -- not part of the public API)
# ---------------------------------------------------------------------------

# Collect every library target defined in `dir`, and in `dir`'s subdirectories
# when RECURSE is given.
#
# This is deliberately a sweep rather than a hand-maintained list. The list it
# would replace is exactly the kind that goes stale silently: a module added to
# src/ and forgotten here would not fail anything, it would just be missing from
# the package, and the first report would come from a consumer. Reading the
# buildsystem back means "installed" and "defined under src/" cannot drift.
#
# UTILITY targets (the calaman_compile_time_tests umbrella) and ALIAS targets
# (:: spellings, which are not in BUILDSYSTEM_TARGETS at all) fall out of the
# TYPE filter on their own.
function(_calaman_collect_library_targets dir out_var)
  cmake_parse_arguments(
    _c
    "RECURSE"
    ""
    ""
    ${ARGN}
  )

  set(_found "")
  get_property(
    _targets
    DIRECTORY "${dir}"
    PROPERTY BUILDSYSTEM_TARGETS
  )
  foreach(_target IN LISTS _targets)
    get_target_property(_type ${_target} TYPE)
    if(_type MATCHES "^(STATIC|SHARED|MODULE|OBJECT|INTERFACE)_LIBRARY$")
      list(APPEND _found ${_target})
    endif()
  endforeach()

  if(_c_RECURSE)
    get_property(
      _subdirs
      DIRECTORY "${dir}"
      PROPERTY SUBDIRECTORIES
    )
    foreach(_subdir IN LISTS _subdirs)
      _calaman_collect_library_targets("${_subdir}" _from_subdir RECURSE)
      list(APPEND _found ${_from_subdir})
    endforeach()
  endif()

  set(${out_var}
      "${_found}"
      PARENT_SCOPE
  )
endfunction()

# The installed module-source directory for `target`, mirroring its position
# under src/ -- see rule 2 in this file's header comment.
function(_calaman_module_destination target out_var)
  get_target_property(_source_dir ${target} SOURCE_DIR)
  file(RELATIVE_PATH _relative "${PROJECT_SOURCE_DIR}/src" "${_source_dir}")
  # A target declared directly in src/CMakeLists.txt has an empty relative path
  # and installs its module sources at the MODULEDIR root; everything else
  # mirrors to a subdirectory (solver/, blas/, ...).
  if(_relative STREQUAL "")
    set(_destination "${CALAMAN_INSTALL_MODULEDIR}")
  else()
    set(_destination "${CALAMAN_INSTALL_MODULEDIR}/${_relative}")
  endif()
  set(${out_var}
      "${_destination}"
      PARENT_SCOPE
  )
endfunction()

# Install the non-module headers that a target's module units #include by
# relative path, next to the installed module sources -- see rule 3.
#
# Nothing here has to name them: a header sitting in a module target's own
# source directory is, by this project's layout, exactly a header that
# directory's module units include by relative path. Globbing at configure time
# is safe for the same reason it is usually not: the set is not an input to any
# build rule, only to an install rule, so a stale glob costs a reconfigure and
# never a wrong build.
function(_calaman_install_module_adjacent_headers target destination)
  get_target_property(_source_dir ${target} SOURCE_DIR)
  file(GLOB _headers "${_source_dir}/*.h" "${_source_dir}/*.cuh")
  if(_headers)
    install(FILES ${_headers} DESTINATION "${destination}")
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Emit every install rule for the project, plus the CMake package.
#
# Call this from the top-level CMakeLists.txt AFTER every add_subdirectory that
# defines a library, because it reads the buildsystem back to decide what to
# install and a target that does not exist yet cannot be found.
function(calaman_install_package)
  # ONE recursive sweep of src/, because src/ is one tree. This is where this
  # file differs most from the WarpWraps version it came from: there, src/ had a
  # per-backend subtree plus a wrappers subtree, and each was swept
  # separately to keep the non-recursive collection of the neutral layer
  # from double-counting them. Here the backend split is the dependency's, so
  # there is nothing to separate -- and `deps/` is not under src/, so the sweep
  # cannot reach WarpWraps or GoogleTest.
  _calaman_collect_library_targets(
    "${PROJECT_SOURCE_DIR}/src" _targets RECURSE
  )
  list(REMOVE_DUPLICATES _targets)

  # THE PART THAT IS NOT SOLVED YET. WarpWraps is consumed with
  # add_subdirectory, so a target collected above carries WarpWraps targets in
  # its INTERFACE_LINK_LIBRARIES, and install(EXPORT) refuses an export set
  # whose interface names a target that is not exported anywhere. Expect the
  # first real call of this function to fail with exactly that, naming a
  # `wwr.*` target; docs/architecture.md section 2 has the two ways out.
  # CALAMAN_INSTALL defaults OFF until one is chosen, which is why this is a
  # comment rather than a workaround.
  if(NOT _targets)
    message(
      WARNING
        "CALAMAN_INSTALL is ON but no library targets are defined under src/, "
        "so the package would install nothing. Leave it OFF until src/ has a "
        "module."
    )
  endif()

  list(LENGTH _targets _count)
  message(STATUS "Install: ${_count} targets in the calaman package")

  # Each target is installed on its own rather than in one install(TARGETS ...)
  # call, because FILE_SET CXX_MODULES needs a per-target DESTINATION (rule 2).
  # They all name the same EXPORT set, which is what makes them one package.
  foreach(_target IN LISTS _targets)
    # Unify the exported name with the in-tree `::` alias. Set before
    # install(TARGETS ... EXPORT), which is what reads EXPORT_NAME.
    _calaman_export_name(${_target} _export_name)
    set_target_properties(${_target} PROPERTIES EXPORT_NAME "${_export_name}")

    get_target_property(_type ${_target} TYPE)

    if(_type STREQUAL "INTERFACE_LIBRARY")
      # An INTERFACE library has no artifact to install; it is in the export
      # set for its usage requirements, and for the plain reason that targets
      # linking it cannot be exported without it.
      install(TARGETS ${_target} EXPORT calaman-targets)
    else()
      _calaman_module_destination(${_target} _module_dir)
      # cmake-lint: disable=E1122
      # One DESTINATION per artifact kind is how install(TARGETS) is spelled --
      # ARCHIVE, LIBRARY, RUNTIME and FILE_SET each take their own. cmake-lint
      # models the command as a flat argument list and reads the repeats as a
      # duplicated keyword.
      install(
        TARGETS ${_target}
        EXPORT calaman-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
                FILE_SET CXX_MODULES
                DESTINATION "${_module_dir}"
      )
      _calaman_install_module_adjacent_headers(${_target} "${_module_dir}")
    endif()
  endforeach()

  # The .h/.cuh headers that are reached by include path rather than by sitting
  # next to a module source. Installing src/'s shape under include/calaman/ is
  # what makes the include spellings resolve unchanged -- see the
  # INSTALL_INTERFACE include directories on the targets themselves.
  #
  # src/'s whole header shape, mirrored under include/calaman/ so that every
  # include spelling a module unit uses in-tree resolves identically against the
  # install -- root-relative ones ("solver/panel.h") and bare
  # same-directory ones alike. See the INSTALL_INTERFACE include directories
  # on the targets themselves, which are what point a consumer at this root.
  #
  # install(DIRECTORY) rather than a glob because src/ is ONE tree here:
  # there is no per-backend subtree to exclude, so recursion is the correct
  # behaviour rather than a hazard. FILES_MATCHING keeps module sources
  # (.cppm) out -- those are installed per target, with their own
  # destinations, above.
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/src/"
    DESTINATION "${CALAMAN_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.cuh"
  )

  # CXX_MODULES_DIRECTORY is what makes this an installable module package
  # rather than a broken one: without it the export names targets whose module
  # file sets no consumer can see, and `import calaman.wrappers.blas;` fails to
  # resolve against a package that otherwise installed cleanly.
  install(
    EXPORT calaman-targets
    FILE calaman-targets.cmake
    NAMESPACE calaman::
    DESTINATION "${CALAMAN_INSTALL_CMAKEDIR}"
    CXX_MODULES_DIRECTORY cxx-modules
  )

  # SameMinorVersion, not SameMajorVersion: at 0.x there is no major-version
  # promise to make, and C++23 module packages have a narrower compatibility
  # story than ordinary libraries anyway (see calamanConfig.cmake.in).
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/calamanConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
  )

  # Recorded into the config so a consumer's build is checked against the
  # configuration this package was actually built with, rather than discovering
  # the mismatch as a link error or, worse, a wrong warp size at runtime.
  set(CALAMAN_PACKAGE_BACKEND "${CALAMAN_GPU_BACKEND}")
  set(CALAMAN_PACKAGE_WARP_SIZE "${CALAMAN_WARP_SIZE}")
  set(CALAMAN_PACKAGE_CXX_COMPILER_ID "${CMAKE_CXX_COMPILER_ID}")
  set(CALAMAN_PACKAGE_CXX_COMPILER_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
  set(CALAMAN_PACKAGE_CXX_STANDARD_LIBRARY "${CMAKE_CXX_STANDARD_LIBRARY}")

  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/calamanConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/calamanConfig.cmake"
    INSTALL_DESTINATION "${CALAMAN_INSTALL_CMAKEDIR}"
  )

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/calamanConfig.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/calamanConfigVersion.cmake"
          DESTINATION "${CALAMAN_INSTALL_CMAKEDIR}"
  )
endfunction()
