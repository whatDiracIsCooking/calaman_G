# ==========================================================================
# GoogleTest
# ==========================================================================
# Fetched and built from source rather than found. The whole project compiles
# and links against libc++ -- `-stdlib=libc++` is a global link option set in
# the root CMakeLists -- so a distro GoogleTest built against libstdc++ would
# not link. Building it here means it inherits those flags.
#
# INSTALL_GTEST OFF keeps `gpumod` installing nothing, which is true today and
# worth preserving: quietly installing GoogleTest's headers and CMake package
# files onto consumers would undo that.
include(FetchContent)

set(WWR_GOOGLETEST_TAG
    "v1.17.0"
    CACHE STRING "googletest git tag to build against"
)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG ${WWR_GOOGLETEST_TAG}
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
#
# That last part is why this project needs no patched GoogleTest. One common
# approach -- a 24KB patch swapping GoogleTest's #includes for `import std;` --
# applies it behind GTEST_BUILD_USING_STD_MODULE set PRIVATE, so only
# GoogleTest's own TUs ever see it and consuming TUs get the ordinary header
# build regardless. Building with headers here puts consumers in exactly that
# same position, with no patch to re-roll on every GoogleTest upgrade.
foreach(_gt_target IN ITEMS gtest gtest_main gmock gmock_main)
  if(TARGET ${_gt_target})
    set_target_properties(${_gt_target} PROPERTIES CXX_SCAN_FOR_MODULES OFF)
  endif()
endforeach()
