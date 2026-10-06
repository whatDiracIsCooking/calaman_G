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
# A build targets exactly one GPU backend: CUDA or HIP. Decided before
# project() because it decides whether the CUDA language is enabled at all --
# a HIP build needs no CUDA toolkit, and a CUDA build needs no ROCm.
#
# THIS PROJECT HAS NO PER-BACKEND SOURCE TREE. The backend split lives in the
# dependency: WarpWraps maps its backend-neutral `wwr*` names onto
# whichever vendor was selected, and everything under src/ here is written once
# against those names. So this variable's only jobs are to pick the languages
# to enable, to pick which vendor packages to find, and to be forwarded to
# WarpWraps as WWR_GPU_BACKEND below.
set(CALAMAN_GPU_BACKEND
    "CUDA"
    CACHE STRING "GPU backend to build against: CUDA or HIP"
)
set_property(CACHE CALAMAN_GPU_BACKEND PROPERTY STRINGS CUDA HIP)
if(NOT CALAMAN_GPU_BACKEND MATCHES "^(CUDA|HIP)$")
  message(
    FATAL_ERROR
      "CALAMAN_GPU_BACKEND must be CUDA or HIP, got '${CALAMAN_GPU_BACKEND}'"
  )
endif()

if(CALAMAN_GPU_BACKEND STREQUAL "CUDA")
  set(_calaman_languages CXX CUDA)
else()
  set(_calaman_languages CXX)
endif()

project(
  calaman
  VERSION 0.1.0
  LANGUAGES ${_calaman_languages}
  DESCRIPTION
    "GPU-accelerated dense linear algebra and solvers in C++23 named modules"
)
message(STATUS "GPU backend: ${CALAMAN_GPU_BACKEND}")

# ========================================================================
# Warp / wavefront size
# ========================================================================
# A configure-time constant because neither backend offers a compile-time
# spelling that is correct in every pass; WarpWraps's runtime.cuh has the full
# account. 32 is right for every NVIDIA arch and for RDNA (gfx10xx-gfx12xx);
# set 64 for CDNA (gfx90a, gfx942). Nothing validates it against a real device.
#
# Forwarded to WarpWraps as WWR_WARP_SIZE below, so there is ONE knob, not
# two that can disagree.
set(CALAMAN_WARP_SIZE
    "32"
    CACHE STRING "Warp/wavefront size: 32 for NVIDIA and RDNA, 64 for CDNA"
)
set_property(CACHE CALAMAN_WARP_SIZE PROPERTY STRINGS 32 64)
if(NOT CALAMAN_WARP_SIZE MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "CALAMAN_WARP_SIZE must be a positive integer, got "
                      "'${CALAMAN_WARP_SIZE}'"
  )
endif()
message(STATUS "GPU warp size: ${CALAMAN_WARP_SIZE}")

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
# Compiler cache
# ========================================================================
# sccache rather than ccache, for a measured reason: a module interface writes
# TWO outputs from one invocation (the object and the BMI), and ccache 4.9
# declares that whole shape uncacheable -- so it cannot cache a single .cppm in
# this tree. sccache 0.18 knows -fmodule-output names a second output and
# restores both. See docs/architecture.md section 7 for the measurement.
#
# Found here rather than pinned in a preset because it is OPTIONAL: CI's hosted
# runners have no sccache, and a launcher baked into the `base` preset would
# fail them at the first compile instead of simply not caching. Set at the top
# level before add_subdirectory(deps), so WarpWraps and GoogleTest cache too.
#
# SCCACHE_BASEDIRS is deliberately set NOWHERE in this repo -- a restored BMI
# keeps the absolute path it was built under, so normalising keys across
# checkouts mounted at different paths buys a broken build rather than a hit.
option(CALAMAN_COMPILER_CACHE "Use sccache as the compiler launcher when found"
       ON
)
if(CALAMAN_COMPILER_CACHE)
  find_program(CALAMAN_SCCACHE sccache)
  if(CALAMAN_SCCACHE)
    set(CMAKE_C_COMPILER_LAUNCHER "${CALAMAN_SCCACHE}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CALAMAN_SCCACHE}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CALAMAN_SCCACHE}")
    message(STATUS "Compiler cache: ${CALAMAN_SCCACHE}")
  else()
    message(STATUS "Compiler cache: sccache not found; no compile is cached")
  endif()
endif()

# ========================================================================
# Find the backend's packages
# ========================================================================
# These are the same calls WarpWraps's own top-level CMakeLists.txt makes, and
# repeating them here is deliberate rather than redundant: an IMPORTED target
# created by find_package is visible only in the directory that found it and
# its subdirectories. WarpWraps is added as a SUBDIRECTORY below, so targets it
# finds (hip::device, roc::hipsolver, ...) are not visible in src/ or test/
# here -- a target this project links transitively through a WarpWraps module
# has to be resolvable in the directory doing the linking. Finding them at the
# top level puts them in scope for the whole tree, WarpWraps included, and the
# second find_package call inside WarpWraps is then a cached no-op.
#
# Keep this list in step with deps/WarpWraps/CMakeLists.txt; a divergence
# surfaces as a "target not found" at generate time, naming a vendor target.
if(CALAMAN_GPU_BACKEND STREQUAL "CUDA")
  find_package(CUDAToolkit 13.0.0 REQUIRED)
  # nvCOMP ships its own config package (nvcomp::nvcomp), linked by
  # wwr.cuda.nvcomp -- not a FindCUDAToolkit component. docker/install-cuda.sh
  # publishes it under /opt/nvidia/nvcomp. cuTENSOR and NCCL are found by hand
  # inside WarpWraps's own src/cuda, so they need no find_package here -- only
  # presence in the image.
  find_package(nvcomp CONFIG REQUIRED)
else()
  # Seed GPU_TARGETS from the environment before find_package(hip).
  #
  # hip-config-amd.cmake reads the CMake variable GPU_TARGETS; the HIP images
  # export an env var of the same name (default gfx1200), but an env var is not
  # a CMake variable, so it must be copied in. Left unset, hip-config runs
  # `amdgpu-arch` and on a box with no AMD GPU silently falls back to gfx906 --
  # wave64, which also disagrees with CALAMAN_WARP_SIZE's default 32.
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
  # rocm_smi is gone on purpose: WarpWraps dropped wwr.hip.rocm_smi, leaving
  # amd_smi as the sole SMI module, so nothing here finds rocm_smi anymore.
  find_package(amd_smi CONFIG REQUIRED)
  find_package(hiprtc CONFIG REQUIRED)
  find_package(hipblas CONFIG REQUIRED)
  find_package(hipblaslt CONFIG REQUIRED)
  find_package(hipsolver CONFIG REQUIRED)
  find_package(hipsparse CONFIG REQUIRED)
  find_package(hipfft CONFIG REQUIRED)
  find_package(hiprand CONFIG REQUIRED)
  # rccl provides roc::rccl (wwr.hip.rccl); hipcomp provides hipcomp::hipcomp
  # (wwr.hip.hipcomp), built from source into /opt/rocm-ds by
  # docker/install-rocm-ds.sh. hipTensor is found by hand inside WarpWraps's
  # own src/hip, so it needs no find_package here -- only presence in the image.
  find_package(rccl CONFIG REQUIRED)
  find_package(hipcomp CONFIG REQUIRED)
endif()

# ========================================================================
# The CPU reference LAPACK
# ========================================================================
# Netlib reference LAPACK plus its C interface, LAPACKE, used as the ORACLE the
# tests compare a GPU factorisation against. Reference rather than OpenBLAS on
# purpose: an oracle has to be deterministic, and a threaded BLAS is not
# bitwise reproducible across thread counts. docker/Dockerfile.base installs
# liblapack-dev + liblapacke-dev, so it is present in every image.
#
# Tests only -- nothing under src/ may link it. A GPU library that quietly
# falls back to a CPU LAPACK is a different library, and the tests could then
# no longer tell the two apart.
option(CALAMAN_LAPACK_REFERENCE
       "Find CPU LAPACK/LAPACKE and expose calaman::lapack_reference for tests"
       ON
)
if(CALAMAN_LAPACK_REFERENCE)
  find_package(LAPACK)
  find_path(CALAMAN_LAPACKE_INCLUDE_DIR NAMES lapacke.h)
  find_library(CALAMAN_LAPACKE_LIBRARY NAMES lapacke)

  if(LAPACK_FOUND
     AND CALAMAN_LAPACKE_INCLUDE_DIR
     AND CALAMAN_LAPACKE_LIBRARY
  )
    # An INTERFACE target rather than bare variables, so a test links one name
    # and the include directory travels with it.
    add_library(calaman_lapack_reference INTERFACE)
    add_library(calaman::lapack_reference ALIAS calaman_lapack_reference)
    target_include_directories(
      calaman_lapack_reference SYSTEM
      INTERFACE "${CALAMAN_LAPACKE_INCLUDE_DIR}"
    )
    target_link_libraries(
      calaman_lapack_reference INTERFACE "${CALAMAN_LAPACKE_LIBRARY}"
                                         LAPACK::LAPACK
    )
    message(
      STATUS "CPU reference LAPACK: ${CALAMAN_LAPACKE_LIBRARY} (LAPACKE) + "
             "${LAPACK_LIBRARIES}"
    )
  else()
    # A WARNING, not an error: the library builds without an oracle, only the
    # tests that need one cannot. Those tests must check for the target rather
    # than assume it, so that this stays a missing TIER and never a silent pass.
    message(
      WARNING
        "CPU reference LAPACK not found (LAPACK_FOUND=${LAPACK_FOUND}, "
        "lapacke.h=${CALAMAN_LAPACKE_INCLUDE_DIR}, "
        "lapacke=${CALAMAN_LAPACKE_LIBRARY}). Tests that compare against a CPU "
        "reference will not be built. Install liblapack-dev and liblapacke-dev,"
        " or configure with -DCALAMAN_LAPACK_REFERENCE=OFF to say so."
    )
  endif()
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
include(cmake/calaman_internal_helpers.cmake)
include(cmake/calaman_add_cxx_module_library.cmake)
include(cmake/calaman_add_gpu_device_library.cmake)
include(cmake/calaman_add_interface_library.cmake)

# Creating ctest-registered plain (non-GoogleTest) test executables
include(cmake/calaman_add_test_executable.cmake)

# Registers GoogleTest binaries with ctest one entry per suite
include(cmake/calaman_add_gtest_suite_tests.cmake)

# Consolidates GoogleTest executable boilerplate (module properties, the shared
# GPU Environment's main.cpp/link list, libstdc++)
include(cmake/calaman_add_gtest_executable.cmake)

# Install rules, the export set and the CMake package. Included HERE rather than
# next to calaman_install_package()'s call at the bottom of this file, because
# it defines the CALAMAN_INSTALL_*DIR variables that src/ and the library macros
# reference in their $<INSTALL_INTERFACE:> include directories -- those have to
# be set before the add_subdirectory calls that read them.
include(cmake/calaman_install.cmake)

# ========================================================================
# AddressSanitizer (host code only)
# ========================================================================
# CALAMAN_ENABLE_ASAN, not a bare ENABLE_ASAN: WarpWraps renamed its own option
# to WWR_ENABLE_ASAN (its #129) and still honours a bare ENABLE_ASAN with a
# deprecation warning, so sharing that name turned its own ASAN on too and
# double-applied the flags. The add_compile_options below is set at the top
# level, before add_subdirectory(deps), so it already reaches WarpWraps's
# targets -- we deliberately do NOT set WWR_ENABLE_ASAN, which would add
# -fsanitize=address a second time.
option(CALAMAN_ENABLE_ASAN
       "Enable AddressSanitizer instrumentation for host (CXX) code" OFF
)
if(CALAMAN_ENABLE_ASAN)
  message(STATUS "AddressSanitizer enabled (host code only)")
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
  )
  add_link_options(-fsanitize=address)
endif()

# ========================================================================
# NVIDIA compute-sanitizer (CUDA only)
# ========================================================================
# Not a compile-time sanitizer: it instruments at RUN time, so the build only
# adds -lineinfo (so a report names a source line) and wraps every target-run
# ctest entry in compute-sanitizer through CMAKE_TEST_LAUNCHER. Wrapping per
# test rather than around ctest keeps labels, -R and per-suite verdicts, and
# needs no --target-processes. AMD has no counterpart, so HIP refuses it.
option(CALAMAN_COMPUTE_SANITIZER
       "Run each ctest entry under NVIDIA compute-sanitizer (CUDA only)" OFF
)
set(CALAMAN_COMPUTE_SANITIZER_TOOL
    "memcheck"
    CACHE STRING "compute-sanitizer --tool (memcheck, racecheck, ...)"
)
set_property(
  CACHE CALAMAN_COMPUTE_SANITIZER_TOOL PROPERTY STRINGS memcheck racecheck
                                                initcheck synccheck
)
if(CALAMAN_COMPUTE_SANITIZER)
  if(NOT CALAMAN_GPU_BACKEND STREQUAL "CUDA")
    message(FATAL_ERROR "CALAMAN_COMPUTE_SANITIZER needs the CUDA backend")
  endif()
  find_program(
    CALAMAN_COMPUTE_SANITIZER_EXECUTABLE
    NAMES compute-sanitizer
    HINTS "${CUDAToolkit_BIN_DIR}"
    REQUIRED
  )
  message(
    STATUS
      "compute-sanitizer enabled (${CALAMAN_COMPUTE_SANITIZER_TOOL}) for ctest"
  )
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  # --require-cuda-init no: host-only suites (arg checks, plans, the compile-
  # time tier) never touch CUDA, which the tool otherwise reports as an error.
  set(CMAKE_TEST_LAUNCHER
      "${CALAMAN_COMPUTE_SANITIZER_EXECUTABLE}" --tool
      "${CALAMAN_COMPUTE_SANITIZER_TOOL}" --error-exitcode 1
      --require-cuda-init no
  )
  # racecheck and synccheck judge a kernel's own barriers and shared memory,
  # so they are scoped to ours (every one mangles `calaman` in). Unscoped,
  # racecheck runs >15x slower in cuBLAS/cuSOLVER and synccheck flags
  # cuBLAS's own Strsm. memcheck/initcheck stay unscoped: a bad argument we
  # pass faults inside the vendor's kernel.
  if(CALAMAN_COMPUTE_SANITIZER_TOOL MATCHES "^(racecheck|synccheck)$")
    list(APPEND CMAKE_TEST_LAUNCHER --kernel-name kns=calaman)
  endif()
endif()

# Scales every TIMEOUT the calaman_add_* test macros set; a sanitized run is
# many times slower than the timeouts were sized for.
set(CALAMAN_TEST_TIMEOUT_MULTIPLIER
    "1"
    CACHE STRING "Integer factor applied to calaman test TIMEOUTs"
)

# ========================================================================
# Coverage (clang source-based, host code only)
# ========================================================================
# Instruments CXX -- and only CXX, like ASAN above -- so nvcc's .cu units are
# left alone. The .profraw the instrumented binaries drop is merged by
# llvm-profdata and read by llvm-cov, both of which must match the clang
# version; devtools/coverage.sh drives that post-processing and the 'coverage'
# preset turns this on in build-coverage/.
option(
  CALAMAN_COVERAGE
  "Enable clang source-based coverage instrumentation for host (CXX) code" OFF
)
if(CALAMAN_COVERAGE)
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
# cmake/calaman_install.cmake).
#
# OFF BY DEFAULT, WHICH IS NOT WHERE THIS SHOULD END UP. WarpWraps is consumed
# with add_subdirectory (see deps/CMakeLists.txt), so a calaman library target
# carries WarpWraps targets in its INTERFACE_LINK_LIBRARIES -- install(EXPORT)
# refuses an export set whose interface names a target that is not exported
# anywhere. Making this work needs a decision that cannot be made before the
# first module exists:
#
#   * install WarpWraps alongside (forward WWR_INSTALL=ON, and have
#     calamanConfig.cmake find_dependency(wwr)) -- two packages, one prefix;
#     needs WarpWraps to offer `wwr::` ALIAS targets in-tree, since that is the
#     spelling an installed consumer resolves; or
#   * consume an already-installed WarpWraps through find_package instead of
#     add_subdirectory, for install builds only.
#
# Until then `devtools/install-check.sh` and CI's install-check job have
# nothing to prove. Turning this ON today will fail at generate time, and that
# failure is the accurate report.
option(CALAMAN_INSTALL "Generate install rules and the CMake package" OFF)

# Some tests are proved by building: static_asserts on type traits, constexpr
# values and enumerator values, checked as the test TU compiles.
#
#   cmake --build build --target calaman_compile_time_tests
#   ctest --test-dir build -L compile_time
#
# With CALAMAN_COMPILE_TIME_ONLY the configure step also skips everything that
# exists only for the runtime tests -- the GoogleTest fetch (deps/) and the
# runtime test directories -- so it needs no network and generates no runtime
# test targets. The 'compile-time' preset turns it on in its own build
# directory.
option(
  CALAMAN_COMPILE_TIME_ONLY
  "Configure only the compile-time tests; skip GoogleTest and the runtime tests"
  OFF
)

# The experimental/ tier -- spikes and sketches not yet promoted to src/. OFF by
# default so a default build (and CI) never sees it; turn it ON to configure the
# experimental modules and their suites.
option(CALAMAN_BUILD_EXPERIMENTAL "Build the experimental/ tier (unstable)" OFF)

# Umbrella target for the compile-time tier. Each compile-time test directory
# add_dependencies()'s its own targets onto it, so it is defined before those
# directories are added.
#
# NAMED calaman_compile_time_tests, not compile_time_tests: WarpWraps defines a
# target of that second name, and two targets cannot share one name in a build.
add_custom_target(
  calaman_compile_time_tests COMMENT "Building the compile-time tests"
)

# Umbrella over every calaman_add_gpu_device_library() target, which registers
# itself here -- what `cross-backend-check.sh --device-only` builds, so the list
# of `.cu` libraries is read from the buildsystem rather than kept by hand.
add_custom_target(
  calaman_device_libraries COMMENT "Building the device-kernel libraries"
)

# ========================================================================
# Enable testing
# ========================================================================
enable_testing()

# ========================================================================
# Subdirectories
# ========================================================================
# WarpWraps (the GPU API layer this project is written against) and GoogleTest.
# First, because everything below links what it defines.
add_subdirectory(deps)

# src/ holds the shipped modules -- one backend-neutral tree written against
# WarpWraps's `wwr*` names (NOT a per-backend subtree), so a module is added
# once and builds for either backend. The first module (calaman.lacpy) lives
# there now; add each new one to src/CMakeLists.txt.
add_subdirectory(src)

# The in-tree examples, written the way a consumer would write them.
#add_subdirectory(example)

# The experimental/ tier, gated off by default (see CALAMAN_BUILD_EXPERIMENTAL
# above). Added after src/ so an experimental module may link a shipped one.
if(CALAMAN_BUILD_EXPERIMENTAL)
  add_subdirectory(experimental)
endif()

# Every suite under test/ is a GoogleTest runtime suite, and GoogleTest is not
# fetched under CALAMAN_COMPILE_TIME_ONLY (deps/), so the two guards move
# together: without this one, configure fails on a missing GTest::gtest_main.
if(NOT CALAMAN_COMPILE_TIME_ONLY)
  add_subdirectory(test)
endif()

# ========================================================================
# Install rules and the CMake package
# ========================================================================
# Last in the file on purpose: calaman_install_package() decides what to
# install by reading the buildsystem back (every library target defined under
# src/), rather than from a hand-maintained list that a new module could fall
# out of silently. A target that has not been defined yet cannot be found, so
# this has to run after every add_subdirectory that defines one.
if(CALAMAN_INSTALL)
  calaman_install_package()
endif()
