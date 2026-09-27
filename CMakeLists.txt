cmake_minimum_required(VERSION 4.2)

# `import std` is still gated behind an experimental feature flag (still
# experimental as of CMake 4.4), and CMake accepts a different gate UUID per
# release ON PURPOSE, to force re-acknowledgement of the unstable API on each
# upgrade. The accepted value lives only inside the cmake binary -- it cannot be
# derived at configure time -- so it must be pinned to the version it belongs
# to. The failure mode is silent: a stale UUID is IGNORED, and every `import
# std` then fails to resolve. So gate the UUID on the CMake version and hard-
# error on any version whose value has not been verified here, rather than let
# an upgrade quietly break module resolution.
#
# When upgrading CMake, find that release's value (Help/dev/experimental.rst, or
# `strings $(command -v cmake)` and match the UUID pattern), confirm it, and add
# a branch below. Verified values:
#   4.2.x  ->  d0edc3af-4c50-42ea-a356-e2862fe7a444   (confirmed against 4.2.3)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.2 AND CMAKE_VERSION VERSION_LESS 4.3)
  set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
else()
  message(
    FATAL_ERROR
      "No verified CMAKE_EXPERIMENTAL_CXX_IMPORT_STD gate UUID for CMake "
      "${CMAKE_VERSION}. The UUID is release-specific and a stale one is "
      "ignored silently, breaking every `import std`. Look up this release's "
      "value (Help/dev/experimental.rst, or `strings $(command -v cmake)`), "
      "verify it, and add a branch above."
  )
endif()

# Configure libc++ BEFORE project() so compiler detection uses it.
# CMAKE_CXX_STANDARD_LIBRARY must be set explicitly so CMake looks for
# libc++.modules.json (not libstdc++.modules.json) when resolving import std.
set(CMAKE_CXX_STANDARD_LIBRARY "libc++")
set(CMAKE_CXX_FLAGS_INIT "-stdlib=libc++")

# Set C++ compiler to Clang BEFORE project()
set(CMAKE_CXX_COMPILER "clang++")

# ========================================================================
# Toolchain discovery (must run before project())
# ========================================================================
# C++23 modules need clang-scan-deps, and `import std` needs libc++'s module
# manifest. Debian/Ubuntu install both under a versioned /usr/lib/llvm-<N>/ that
# is not on PATH, so search there, preferring the newest version found. Override
# either if discovery picks the wrong toolchain:
#
#   -DCMAKE_CXX_COMPILER_CLANG_SCAN_DEPS=/path/to/clang-scan-deps
#   -DCMAKE_CXX_STDLIB_MODULES_JSON=/path/to/libc++.modules.json
file(
  GLOB _llvm_roots
  LIST_DIRECTORIES true
  /usr/lib/llvm-* /usr/local/llvm-*
)
list(
  SORT _llvm_roots
  COMPARE NATURAL
  ORDER DESCENDING
)

set(_llvm_bin_hints "")
set(_llvm_lib_hints "")
foreach(_root IN LISTS _llvm_roots)
  list(APPEND _llvm_bin_hints "${_root}/bin")
  list(APPEND _llvm_lib_hints "${_root}/lib")
endforeach()

if(NOT CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS)
  find_program(
    CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS
    NAMES clang-scan-deps
    HINTS ${_llvm_bin_hints}
  )
  if(NOT CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS)
    message(
      FATAL_ERROR
        "clang-scan-deps not found (required for C++23 module scanning). "
        "Install the matching LLVM tools package, or pass "
        "-DCMAKE_CXX_COMPILER_CLANG_SCAN_DEPS=<path>."
    )
  endif()
endif()

if(NOT CMAKE_CXX_STDLIB_MODULES_JSON)
  find_file(
    CMAKE_CXX_STDLIB_MODULES_JSON
    NAMES libc++.modules.json
    HINTS ${_llvm_lib_hints}
    PATHS /usr/lib /usr/local/lib
  )
  if(NOT CMAKE_CXX_STDLIB_MODULES_JSON)
    message(
      FATAL_ERROR
        "libc++.modules.json not found (required for `import std`). "
        "Install libc++-dev built with module support, or pass "
        "-DCMAKE_CXX_STDLIB_MODULES_JSON=<path>."
    )
  endif()
endif()

message(STATUS "Using clang-scan-deps: ${CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS}")
message(
  STATUS "Using libc++ modules manifest: ${CMAKE_CXX_STDLIB_MODULES_JSON}"
)

# ========================================================================
# GPU backend
# ========================================================================
# End users build against exactly one GPU backend: CUDA or HIP. Decided before
# project() because it decides whether the CUDA language is enabled at all.
#
#   CUDA -> src/cuda, test/cuda, and the CUDA toolkit
#   HIP  -> src/hip, test/hip, and ROCm
#
# Either way the gpu* layer maps the gpu* names onto the chosen backend, and
# src/wrappers + test/wrappers are built on top of it, unchanged. A HIP build
# needs no CUDA toolkit, and a CUDA build needs no ROCm.
set(WWR_GPU_BACKEND
    "CUDA"
    CACHE STRING "GPU backend to build against: CUDA or HIP"
)
set_property(CACHE WWR_GPU_BACKEND PROPERTY STRINGS CUDA HIP)
if(NOT WWR_GPU_BACKEND MATCHES "^(CUDA|HIP)$")
  message(
    FATAL_ERROR
      "WWR_GPU_BACKEND must be CUDA or HIP, got '${WWR_GPU_BACKEND}'"
  )
endif()

if(WWR_GPU_BACKEND STREQUAL "CUDA")
  set(_wwr_languages CXX CUDA)
else()
  set(_wwr_languages CXX)
endif()

project(
  wwr
  VERSION 0.1.0
  LANGUAGES ${_wwr_languages}
  DESCRIPTION "C++23 module wrappers for the CUDA and HIP GPU APIs"
)
message(STATUS "GPU backend: ${WWR_GPU_BACKEND}")

# ========================================================================
# Warp / wavefront size
# ========================================================================
# WWR_WARP_SIZE (src/runtime.cuh) is a configure-time constant because
# neither backend offers a compile-time spelling that is correct in every pass.
# 32 is right for every NVIDIA arch and for RDNA (gfx10xx-gfx12xx);
# set 64 for CDNA (gfx90a, gfx942). Nothing validates it against a real device.
set(WWR_WARP_SIZE
    "32"
    CACHE STRING "Warp/wavefront size: 32 for NVIDIA and RDNA, 64 for CDNA"
)
set_property(CACHE WWR_WARP_SIZE PROPERTY STRINGS 32 64)
if(NOT WWR_WARP_SIZE MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "WWR_WARP_SIZE must be a positive integer, got "
                      "'${WWR_WARP_SIZE}'"
  )
endif()
message(STATUS "GPU warp size: ${WWR_WARP_SIZE}")

# ========================================================================
# C++ Standard and Compiler Requirements
# ========================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Note: CMAKE_CXX_EXTENSIONS must match what the std module uses Leave it at
# default (ON) to match libc++ std module compilation
set(CMAKE_CXX_EXTENSIONS ON)

# Require Clang with libc++
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(
    FATAL_ERROR "This project requires Clang compiler for C++23 module support"
  )
endif()

# Add libc++ linker flags for CXX targets
add_link_options($<$<COMPILE_LANGUAGE:CXX>:-stdlib=libc++>)
add_link_options($<$<COMPILE_LANGUAGE:CXX>:-lc++abi>)

# ========================================================================
# Find the backend's packages
# ========================================================================
if(WWR_GPU_BACKEND STREQUAL "CUDA")
  find_package(CUDAToolkit REQUIRED)
else()
  # Seed GPU_TARGETS from the environment before find_package(hip).
  #
  # hip-config-amd.cmake reads the CMake variable GPU_TARGETS; the HIP images
  # export an env var of the same name (default gfx1200), but an env var is not
  # a CMake variable, so it must be copied in. Left unset, hip-config runs
  # `amdgpu-arch` and on a box with no AMD GPU silently falls back to gfx906 --
  # wave64, which also disagrees with WWR_WARP_SIZE's default 32.
  # Precedence: an explicit -DGPU_TARGETS (already cached) beats
  # $ENV{GPU_TARGETS} beats ROCm's own detection. AMDGPU_TARGETS is left unset
  # -- hip-config warns it is deprecated and copies it into GPU_TARGETS anyway.
  if(NOT DEFINED GPU_TARGETS AND DEFINED ENV{GPU_TARGETS})
    set(GPU_TARGETS
        "$ENV{GPU_TARGETS}"
        CACHE STRING "AMD GPU targets to compile for (from the environment)"
    )
    message(STATUS "GPU_TARGETS from environment: ${GPU_TARGETS}")
  endif()

  find_package(hip CONFIG REQUIRED)
  # rocm_smi and amd_smi are their own separate CMake packages (not part of the
  # `hip` package) -- see src/hip/rocm_smi.cppm and src/hip/amd_smi.cppm for why
  # nvml's HIP counterpart is two modules, not one.
  find_package(rocm_smi CONFIG REQUIRED)
  find_package(amd_smi CONFIG REQUIRED)
  # hiprtc (HIP Runtime Compilation) ships as its own ROCm CMake package,
  # separate from `hip` above -- wwr.hip.hiprtc needs the hiprtc::hiprtc
  # imported target it provides. See src/hip/CMakeLists.txt.
  find_package(hiprtc CONFIG REQUIRED)
  # Library-linked module batch -- each of these is its own ROCm CMake package,
  # separate from `hip` above. See src/hip/CMakeLists.txt for which imported
  # target each provides and links.
  find_package(hipblas CONFIG REQUIRED)
  find_package(hipblaslt CONFIG REQUIRED)
  find_package(hipsolver CONFIG REQUIRED)
  find_package(hipsparse CONFIG REQUIRED)
  # hipfft also covers hipfftXt.h -- same package, same imported target;
  # wwr.hip.hipfftXt does not add a second find_package call.
  find_package(hipfft CONFIG REQUIRED)
  find_package(hiprand CONFIG REQUIRED)
endif()

# nvcc (and hipcc) link against libstdc++ internally. When the host linker is
# clang with -stdlib=libc++, the GCC library directory is not in the default
# search path and -lstdc++ fails to resolve. Expose it via link_directories and
# rpath rather than hardcoding a distro-specific .so path.
execute_process(
  COMMAND gcc --print-file-name=libstdc++.so
  OUTPUT_VARIABLE STDCXX_LIBRARY
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _gcc_result
)
if(_gcc_result EQUAL 0 AND NOT STDCXX_LIBRARY STREQUAL "libstdc++.so")
  get_filename_component(STDCXX_DIR "${STDCXX_LIBRARY}" DIRECTORY)
  add_link_options("-L${STDCXX_DIR}")
  list(APPEND CMAKE_BUILD_RPATH "${STDCXX_DIR}")
  message(STATUS "Found libstdc++ directory: ${STDCXX_DIR}")
endif()

# ========================================================================
# Module Configuration
# ========================================================================
include(cmake/wwr_internal_helpers.cmake)
include(cmake/wwr_add_cxx_module_library.cmake)
include(cmake/wwr_add_gpu_device_library.cmake)
include(cmake/wwr_add_interface_library.cmake)

# Creating ctest-registered plain (non-GoogleTest) test executables
include(cmake/wwr_add_test_executable.cmake)

# Registers GoogleTest binaries with ctest one entry per suite
include(cmake/wwr_add_gtest_suite_tests.cmake)

# Consolidates GoogleTest executable boilerplate (module properties, the shared
# GPU Environment's main.cpp/link list, libstdc++)
include(cmake/wwr_add_gtest_executable.cmake)

# Build-time check that an extension module's wrappers dispatch to the vendor
# functions its TOML table names, read out of the compiled objects
include(cmake/wwr_add_dispatch_check.cmake)

# Install rules, the export set and the CMake package. Included HERE rather than
# next to wwr_install_package()'s call at the bottom of this file, because it
# defines the WWR_INSTALL_*DIR variables that src, src/wrappers and the
# library macros reference in their $<INSTALL_INTERFACE:> include directories --
# those have to be set before the add_subdirectory calls that read them.
include(cmake/wwr_install.cmake)

# Common module compilation flags. Empty on purpose, and NOT removable:
# src/cuda/CMakeLists.txt links it from three targets, so deleting it fails the
# link with `cannot find -lwwr_module_flags`. Do not add -fmodules /
# -fimplicit-module-maps / -fbuiltin-module-map here: with clang 20+ they
# collide with std module-map resolution when a CUDA header (e.g. cublas_v2.h)
# pulls a standard header into its global module fragment.
add_library(wwr_module_flags INTERFACE)

# ========================================================================
# AddressSanitizer (host code only)
# ========================================================================
option(ENABLE_ASAN
       "Enable AddressSanitizer instrumentation for host (CXX) code" OFF
)
if(ENABLE_ASAN)
  message(STATUS "AddressSanitizer enabled (host code only)")
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
  )
  add_link_options(-fsanitize=address)
endif()

# ========================================================================
# Coverage (clang source-based, host code only)
# ========================================================================
# Instruments CXX -- and only CXX, like ASAN above -- so nvcc's .cu units are
# left alone: the runtime tier that actually executes is CPU-only host code
# (test/gpu/conversions), which is what coverage can measure. The .profraw the
# instrumented binaries drop is merged by llvm-profdata and read by llvm-cov,
# both of which must match the clang version; devtools/coverage.sh drives that
# post-processing and the 'coverage' preset turns this on in build-coverage/.
option(WWR_COVERAGE
       "Enable clang source-based coverage instrumentation for host (CXX) code"
       OFF
)
if(WWR_COVERAGE)
  message(
    STATUS "Coverage instrumentation enabled (clang source-based, host CXX)"
  )
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-fprofile-instr-generate>
    $<$<COMPILE_LANGUAGE:CXX>:-fcoverage-mapping>
  )
  add_link_options(-fprofile-instr-generate)
endif()

# ========================================================================
# Options
# ========================================================================
# Whether to generate install rules and the CMake package (see
# cmake/wwr_install.cmake). Defaults to ON for a top-level build and OFF when
# this project is pulled into another one with add_subdirectory or FetchContent:
# a consumer embedding gpumod that way links its targets directly and has no use
# for gpumod's install rules, while `make install` in the OUTER project would
# otherwise drag gpumod's headers and module sources into that project's install
# tree uninvited.
option(WWR_INSTALL "Generate install rules and the CMake package"
       ${PROJECT_IS_TOP_LEVEL}
)

# Some tests are proved by building: static_asserts on type traits, constexpr
# values and enumerator values, checked as the test TU compiles. Those live in
# test/cuda, test/hip and test/gpu (the wrappers) and test/extension/build_time
# (the extension modules), and nothing about them needs GoogleTest or a GPU.
#
#   cmake --build build --target compile_time_tests   # build only those
#   ctest --test-dir build -L compile_time            # run the linked binary
#
# With WWR_COMPILE_TIME_ONLY the configure step also skips everything that
# exists only for the runtime tests -- the GoogleTest fetch (deps/) and the
# runtime test directories -- so it needs no network and generates no runtime
# test targets. The 'compile-time' preset turns it on in its own build
# directory.
option(
  WWR_COMPILE_TIME_ONLY
  "Configure only the compile-time tests; skip GoogleTest and the runtime tests"
  OFF
)

# cuda_compile_tests links the driver-API stubs, so the binary needs
# libcuda.so.1 and libnvidia-ml.so.1 at LOAD time even though it never calls
# the driver. With this ON, test/cuda/CMakeLists.txt points that ONE ctest
# entry at the toolkit's own stubs and drops its `gpu` label, so a machine with
# no driver runs it instead of excluding it -- which is what makes the two CI
# legs report the same count. OFF everywhere but the ci-cuda preset.
#
# It is deliberately not a property of the IMAGE: putting stubs on a
# container-wide LD_LIBRARY_PATH would beat the loader cache and shadow the
# real driver the NVIDIA runtime injects. See test/cuda/CMakeLists.txt.
option(
  WWR_CUDA_DRIVER_STUBS
  "Run cuda_compile_tests against the CUDA driver stubs (no driver needed)"
  OFF
)

# Umbrella target. Each compile-time test directory add_dependencies()'s its own
# targets onto it, so it is defined before those directories are added.
add_custom_target(compile_time_tests COMMENT "Building the compile-time tests")

# ========================================================================
# Enable testing
# ========================================================================
enable_testing()

# ========================================================================
# Subdirectories
# ========================================================================
# src/ owns its own internal order: the raw 1:1 header modules for the chosen
# backend, then the gpu* layer over them (which lives directly in src/), then
# the wrappers -- which are backend-neutral in full: src/wrappers is written
# once against the gpu* names and builds for either backend. There is no
# per-backend wrapper tree -- neither src/hip/extension nor src/cuda/extension
# exists. wwr.wrappers.* adapts what both vendors offer, and a vendor-only
# call goes through the
# low-level module (wwr.cuda.cublas_v2, wwr.cuda.cusolverDn) directly.
#add_subdirectory(src)

# The in-tree examples, written against the gpu* and extension layers the same
# way a consumer would write them. Added after src/ so the targets they link
# exist, and unconditionally -- WWR_COMPILE_TIME_ONLY included -- so
# devtools/cross-backend-check.sh compiles them for the other backend. See
# example/README.md; example/consumer is NOT part of this build.
#add_subdirectory(example)

# GoogleTest, which the runtime tests are written in. Added before the test
# directories so the GTest::* targets exist when they are linked. Not needed by
# the compile-time tests, so WWR_COMPILE_TIME_ONLY skips it.
if(NOT WWR_COMPILE_TIME_ONLY)
  add_subdirectory(deps)
endif()

#if(WWR_GPU_BACKEND STREQUAL "CUDA")
#  add_subdirectory(test/cuda)
#else()
#  add_subdirectory(test/hip)
#endif()
## the gpu* layer's compile-time tests: every gpu* name is the backend's own.
#add_subdirectory(test/gpu)
#add_subdirectory(test/wrappers)
## the extension layer's tests: build-time static_asserts plus GoogleTest runtime
## suites (runtime/memory_buffer/rand need a device).
#add_subdirectory(test/extension)

# ========================================================================
# Install rules and the CMake package
# ========================================================================
# Last in the file on purpose: wwr_install_package() decides what to install
# by reading the buildsystem back (every library target defined under src/),
# rather than from a hand-maintained list that a new module could fall out of
# silently. A target that has not been defined yet cannot be found, so this has
# to run after every add_subdirectory that defines one.
if(WWR_INSTALL)
  wwr_install_package()
endif()
