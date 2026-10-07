# ==========================================================================
# WarpWraps -- the GPU API layer this project is built on
# ==========================================================================
# A git submodule (see ../.gitmodules), added as a subdirectory so there is ONE
# build, one toolchain and one backend variable across both projects. Refresh a
# fresh checkout with:
#
#   git submodule update --init --recursive
#
# devtools/doctor.sh warns when it is absent (DOCTOR_REQUIRED_PATHS), because
# the configure error otherwise points at this file rather than at the missing
# checkout.
#
# WHAT IT PROVIDES, and therefore what src/ here may import:
#
#   wwr.cuda.* / wwr.hip.*     the raw 1:1 vendor header modules
#   wwr.blas, wwr.solver,      the backend-neutral layer: one `wwr*` name per
#   wwr.sparse, wwr.fft,       vendor name, mapped onto whichever backend this
#   wwr.rand, wwr.complex,     build selected
#   wwr.fp16, wwr.bf16,
#   wwr.runtime_api, wwr.tx
#   wwr.wrappers.*             type-safe templated wrappers over those
#   wwr.extension.*            handles, error policies, device buffers,
#                              parallel_for, and the convenience layer
#
# THE BACKEND IS DECIDED IN THE TOP-LEVEL CMakeLists.txt AND FORWARDED HERE.
# WarpWraps reads WWR_GPU_BACKEND before its project() call; the values below
# are written as cache entries before add_subdirectory so that read finds them.
# FORCE is deliberate: without it a stale entry from an earlier configure in
# the same build directory would win over the backend this build asked for, and
# the two projects would then compile for different vendors in one tree.
set(WWR_GPU_BACKEND
    "${CALAMAN_GPU_BACKEND}"
    CACHE STRING "GPU backend for WarpWraps (follows CALAMAN_GPU_BACKEND)" FORCE
)
set(WWR_WARP_SIZE
    "${CALAMAN_WARP_SIZE}"
    CACHE STRING "Warp size for WarpWraps (follows CALAMAN_WARP_SIZE)" FORCE
)

# WarpWraps's runtime test suites and examples are its repository's business,
# not this build's -- but its CMakeLists.txt adds them unconditionally, with no
# PROJECT_IS_TOP_LEVEL guard, so the only lever available from here is its
# compile-time-only switch. ON leaves its src/ fully built (which this project
# links) while skipping its GoogleTest fetch and its runtime test directories.
#
# What still gets built and ctest-registered from WarpWraps: its compile-time
# static_assert tier and its examples. That is the cost of add_subdirectory
# consumption today. The clean fix belongs UPSTREAM -- guarding those
# add_subdirectory calls on `PROJECT_IS_TOP_LEVEL OR WWR_BUILD_TESTING` -- and
# once WarpWraps does that, this variable can go back to following
# CALAMAN_COMPILE_TIME_ONLY.
#
# OFF gives you WarpWraps's whole suite inside this build, which is occasionally
# what you want when bisecting a failure across the two projects.
option(CALAMAN_WARPWRAPS_TESTS
       "Configure WarpWraps's own runtime test suites inside this build" OFF
)
if(CALAMAN_WARPWRAPS_TESTS)
  set(WWR_COMPILE_TIME_ONLY
      OFF
      CACHE BOOL "WarpWraps: configure its runtime tests too" FORCE
  )
else()
  set(WWR_COMPILE_TIME_ONLY
      ON
      CACHE BOOL "WarpWraps: skip its GoogleTest fetch and runtime tests" FORCE
  )
endif()

# WarpWraps's own install rules. Its option defaults to ${PROJECT_IS_TOP_LEVEL},
# which is FALSE here, so this is already OFF -- spelled out because the
# unresolved half of CALAMAN_INSTALL (see the top-level CMakeLists.txt) is
# exactly whether this has to become ON.
set(WWR_INSTALL
    OFF
    CACHE BOOL "WarpWraps: generate its own install rules" FORCE
)

add_subdirectory(WarpWraps)

# Under UBSan, WarpWraps's build-time dispatch checks (*_dispatch targets) leave
# ALL: -fsanitize=function adds relocations to each vendor call through a
# WWR_FUNCTION alias, and the check counts relocations. Every other preset and
# CI still build them (docs/sanitizers.md, S6).
function(_calaman_exclude_dispatch_checks _dir)
  get_property(
    _targets
    DIRECTORY ${_dir}
    PROPERTY BUILDSYSTEM_TARGETS
  )
  foreach(_t IN LISTS _targets)
    if(_t MATCHES "_dispatch$")
      set_target_properties(${_t} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    endif()
  endforeach()
  get_property(
    _subdirs
    DIRECTORY ${_dir}
    PROPERTY SUBDIRECTORIES
  )
  foreach(_sub IN LISTS _subdirs)
    _calaman_exclude_dispatch_checks(${_sub})
  endforeach()
endfunction()
if(CALAMAN_ENABLE_UBSAN)
  _calaman_exclude_dispatch_checks(${CMAKE_CURRENT_SOURCE_DIR}/WarpWraps)
endif()

# ==========================================================================
# GoogleTest
# ==========================================================================
# Fetched and built from source rather than found. The whole project compiles
# and links against libc++ -- `-stdlib=libc++` is a global link option set in
# the root CMakeLists -- so a distro GoogleTest built against libstdc++ would
# not link. Building it here means it inherits those flags.
#
# Skipped entirely under CALAMAN_COMPILE_TIME_ONLY: the compile-time tier is
# static_asserts, which need no test framework, so that preset needs no network.
#
# INSTALL_GTEST OFF keeps this project installing nothing of GoogleTest's:
# quietly installing its headers and CMake package files onto consumers would
# make them a dependency of ours.
if(NOT CALAMAN_COMPILE_TIME_ONLY)
  include(FetchContent)

  set(CALAMAN_GOOGLETEST_TAG
      "v1.17.0"
      CACHE STRING "googletest git tag to build against"
  )

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG ${CALAMAN_GOOGLETEST_TAG}
    GIT_SHALLOW TRUE
  )

  set(INSTALL_GTEST
      OFF
      CACHE BOOL "" FORCE
  )
  set(BUILD_GMOCK
      ON
      CACHE BOOL "" FORCE
  )

  FetchContent_MakeAvailable(googletest)

  # GoogleTest's own sources are plain C++: no named modules, no `import std`.
  # Saying so explicitly skips clang-scan-deps on every GoogleTest TU (it would
  # find nothing) and keeps these targets off CXX_MODULE_STD.
  foreach(_gt_target IN ITEMS gtest gtest_main gmock gmock_main)
    if(TARGET ${_gt_target})
      set_target_properties(${_gt_target} PROPERTIES CXX_SCAN_FOR_MODULES OFF)
    endif()
  endforeach()
endif()
