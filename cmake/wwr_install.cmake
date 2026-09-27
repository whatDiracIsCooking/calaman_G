# wwr_install.cmake — install rules, the export set, and the package config.
#
# A C++23 module package ships module interface SOURCES (.cppm), not BMIs (a BMI
# is not portable), and the consumer's own build recompiles them. Three rules
# follow, each enforced below and explained in full in cmake/README.md, "Install
# and the CMake package":
#
#   1. Every compile requirement of a .cppm must reach the consumer — a PRIVATE
#      requirement is not exported, so it becomes a broken install.
#   2. Module sources need per-target destinations (six modules are each rooted
#      at a file named interface.cppm; one shared dir would collide).
#   3. A header a module unit #includes is installed next to those sources.
#
# The consumer-facing half of the contract (same compiler, standard library and
# backend) is stated and enforced in wwrConfig.cmake.in.

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Where the pieces of the installed package live, all relative to the install
# prefix. `modules` holds module interface SOURCES, not BMIs (see above); it is
# under include/ because that is where a "things the consumer's compiler reads"
# directory belongs, not because anything #includes from it.
set(WWR_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/wwr")
set(WWR_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/wwr")
set(WWR_INSTALL_MODULEDIR "${WWR_INSTALL_INCLUDEDIR}/modules")

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
# UTILITY targets (the compile_time_tests umbrella) and ALIAS targets (the ::
# spellings, which are not in BUILDSYSTEM_TARGETS at all) fall out of the TYPE
# filter on their own.
function(_wwr_collect_library_targets dir out_var)
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
      _wwr_collect_library_targets("${_subdir}" _from_subdir RECURSE)
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
function(_wwr_module_destination target out_var)
  get_target_property(_source_dir ${target} SOURCE_DIR)
  file(RELATIVE_PATH _relative "${PROJECT_SOURCE_DIR}/src" "${_source_dir}")
  # The gpu* layer lives directly in src/, so its relative path is empty and its
  # module sources install at the MODULEDIR root; everything else mirrors to a
  # subdirectory (cuda/, wrappers/blas, ...).
  if(_relative STREQUAL "")
    set(_destination "${WWR_INSTALL_MODULEDIR}")
  else()
    set(_destination "${WWR_INSTALL_MODULEDIR}/${_relative}")
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
function(_wwr_install_module_adjacent_headers target destination)
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
function(wwr_install_package)
  # The backend's own wrapper directory is the only part of src/ that is not
  # added unconditionally, so the sweep covers whichever one this build chose.
  if(WWR_GPU_BACKEND STREQUAL "CUDA")
    set(_backend_dir "${PROJECT_SOURCE_DIR}/src/cuda")
  else()
    set(_backend_dir "${PROJECT_SOURCE_DIR}/src/hip")
  endif()

  _wwr_collect_library_targets("${_backend_dir}" _backend_targets RECURSE)
  # The gpu* layer is defined directly in src/CMakeLists.txt, so it is collected
  # from src/ NON-recursively; recursing would re-collect src/cuda, src/hip and
  # src/wrappers, which are swept separately above and below.
  _wwr_collect_library_targets("${PROJECT_SOURCE_DIR}/src" _gpu_targets)
  _wwr_collect_library_targets(
    "${PROJECT_SOURCE_DIR}/src/wrappers" _wrapper_targets RECURSE
  )

  # wwr_module_flags is defined in the top-level CMakeLists.txt rather than
  # under src/, so the sweep above does not reach it -- but three src/cuda
  # targets link it PUBLIC, which puts it in their INTERFACE_LINK_LIBRARIES and
  # so makes install(EXPORT) refuse the whole export set until it is a member
  # too. Collected non-recursively: recursing from the top would pull in test/
  # and deps/ (GoogleTest), none of which belongs in this package.
  _wwr_collect_library_targets("${PROJECT_SOURCE_DIR}" _root_targets)
  list(
    FILTER
    _root_targets
    INCLUDE
    REGEX
    "^wwr_module_flags$"
  )

  set(_targets ${_root_targets} ${_backend_targets} ${_gpu_targets}
               ${_wrapper_targets}
  )
  list(REMOVE_DUPLICATES _targets)

  list(LENGTH _targets _count)
  message(STATUS "Install: ${_count} targets in the wwr package")

  # Each target is installed on its own rather than in one install(TARGETS ...)
  # call, because FILE_SET CXX_MODULES needs a per-target DESTINATION (rule 2).
  # They all name the same EXPORT set, which is what makes them one package.
  foreach(_target IN LISTS _targets)
    get_target_property(_type ${_target} TYPE)

    if(_type STREQUAL "INTERFACE_LIBRARY")
      # An INTERFACE library has no artifact to install; it is in the export
      # set for its usage requirements, and for the plain reason that targets
      # linking it cannot be exported without it.
      install(TARGETS ${_target} EXPORT wwr-targets)
    else()
      _wwr_module_destination(${_target} _module_dir)
      # cmake-lint: disable=E1122
      # One DESTINATION per artifact kind is how install(TARGETS) is spelled --
      # ARCHIVE, LIBRARY, RUNTIME and FILE_SET each take their own. cmake-lint
      # models the command as a flat argument list and reads the repeats as a
      # duplicated keyword.
      install(
        TARGETS ${_target}
        EXPORT wwr-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
                FILE_SET CXX_MODULES
                DESTINATION "${_module_dir}"
      )
      _wwr_install_module_adjacent_headers(${_target} "${_module_dir}")
    endif()
  endforeach()

  # The .h/.cuh headers that are reached by include path rather than by sitting
  # next to a module source. Installing src/'s shape under include/wwr/ is
  # what makes the include spellings resolve unchanged -- see the
  # INSTALL_INTERFACE include directories on the targets themselves.
  #
  # The gpu* layer's backend-switch headers (gpu_backend.h, selected_backend.h,
  # device_guard.h) and its device-side .cuh headers are included bare (e.g.
  # "gpu_backend.h") and live directly in src/, so a NON-recursive glob is
  # exactly this set and they install at the include root. install(DIRECTORY)
  # is wrong here: it would recurse into src/cuda, src/hip and src/wrappers,
  # whose headers are handled separately (wrappers', below; the backends have
  # none). A configure-time glob is safe here: the set is an input only to an
  # install rule, never to a build rule -- a stale glob costs a reconfigure, not
  # a wrong build.
  file(GLOB _gpu_layer_headers "${PROJECT_SOURCE_DIR}/src/*.h"
       "${PROJECT_SOURCE_DIR}/src/*.cuh"
  )
  install(FILES ${_gpu_layer_headers}
          DESTINATION "${WWR_INSTALL_INCLUDEDIR}"
  )

  # The shared dispatch header under src/wrappers, included through src/ (e.g.
  # "wrappers/common/dispatch_sdcz.h"), mirrored to include/wwr/wrappers.
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/src/wrappers/"
    DESTINATION "${WWR_INSTALL_INCLUDEDIR}/wrappers"
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.cuh"
  )

  # The extension layer (src/extension) is built in-tree but is NOT part of the
  # installed package -- the target sweep above collects only the backend dir,
  # the gpu* layer and src/wrappers. Its src/extension/bridge/ headers are
  # #included solely by extension module units (wwr.extension.init_state,
  # wwr.extension.random_normal) and by parallel_for.cuh, so with no
  # extension target exported nothing in the package includes them and there is
  # nothing to install. When the layer is made installable (a
  # WWR_BUILD_EXTENSION
  # opt-in), the rule that installs these headers belongs there, next to the
  # sweep that adds the targets that need them -- so install-check can actually
  # verify it.

  # CXX_MODULES_DIRECTORY is what makes this an installable module package
  # rather than a broken one: without it the export names targets whose module
  # file sets no consumer can see, and `import wwr.wrappers.blas;` fails to
  # resolve against a package that otherwise installed cleanly.
  install(
    EXPORT wwr-targets
    FILE wwr-targets.cmake
    NAMESPACE wwr::
    DESTINATION "${WWR_INSTALL_CMAKEDIR}"
    CXX_MODULES_DIRECTORY cxx-modules
  )

  # SameMinorVersion, not SameMajorVersion: at 0.x there is no major-version
  # promise to make, and C++23 module packages have a narrower compatibility
  # story than ordinary libraries anyway (see wwrConfig.cmake.in).
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/wwrConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
  )

  # Recorded into the config so a consumer's build is checked against the
  # configuration this package was actually built with, rather than discovering
  # the mismatch as a link error or, worse, a wrong warp size at runtime.
  set(WWR_PACKAGE_BACKEND "${WWR_GPU_BACKEND}")
  set(WWR_PACKAGE_WARP_SIZE "${WWR_WARP_SIZE}")
  set(WWR_PACKAGE_CXX_COMPILER_ID "${CMAKE_CXX_COMPILER_ID}")
  set(WWR_PACKAGE_CXX_COMPILER_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
  set(WWR_PACKAGE_CXX_STANDARD_LIBRARY "${CMAKE_CXX_STANDARD_LIBRARY}")

  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/wwrConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/wwrConfig.cmake"
    INSTALL_DESTINATION "${WWR_INSTALL_CMAKEDIR}"
  )

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/wwrConfig.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/wwrConfigVersion.cmake"
          DESTINATION "${WWR_INSTALL_CMAKEDIR}"
  )
endfunction()
