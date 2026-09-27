# A standalone project that consumes an INSTALLED gpumod.
#
# Deliberately not part of the main build: the whole point is that it knows
# nothing about gpumod's source tree and reaches it only through
# find_package(wwr). devtools/install-check.sh builds it that way -- install
# gpumod to a throwaway prefix, then configure this against that prefix -- which
# is the only way an install rule can actually be verified rather than assumed.
#
# Everything above project() is the toolchain setup a consumer needs too. It is
# not gpumod boilerplate leaking out; it is the consequence of consuming a C++23
# module package. The consumer's build COMPILES gpumod's module interface
# sources (a BMI is not portable, so the package ships sources -- see
# cmake/wwr_install.cmake in the gpumod tree), and those sources `import
# std;`. So the flags that make `import std` work have to be set here, before
# project(), exactly as gpumod's own top-level CMakeLists.txt sets them.

cmake_minimum_required(VERSION 4.2)

# The experimental UUID for `import std`, tied to the CMake version -- look the
# current value up in that release's Help/dev/experimental.rst. A stale UUID is
# ignored silently and every `import std` then fails to resolve.
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")

set(CMAKE_CXX_STANDARD_LIBRARY "libc++")
set(CMAKE_CXX_FLAGS_INIT "-stdlib=libc++")
set(CMAKE_CXX_COMPILER "clang++")

project(wwr_consumer_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

add_link_options($<$<COMPILE_LANGUAGE:CXX>:-stdlib=libc++>)
add_link_options($<$<COMPILE_LANGUAGE:CXX>:-lc++abi>)

# The one line that is actually about gpumod.
#
# CONFIG mode: gpumod ships a package config (wwrConfig.cmake), not a Find
# module. Point CMAKE_PREFIX_PATH at the install prefix to locate it.
find_package(wwr 0.1 REQUIRED)

message(STATUS "wwr backend:   ${WWR_GPU_BACKEND}")
message(STATUS "wwr warp size: ${WWR_WARP_SIZE}")

add_executable(consumer main.cpp)

# The :: spellings are the exported names. Linking them brings in gpumod's
# module sources, which this build compiles -- `import wwr.wrappers.blas;`
# in main.cpp resolves because of these lines and nothing else.
#
# The consumer imports two gpu* modules directly (wwr.runtime_api for device
# memory, wwr.blas for the handle), so it names them here. The wrapper
# targets each LINK_PUBLIC the gpu* module they wrap, so those names would
# arrive transitively too, but the example uses runtime_api/blas in its own
# right and so depends on them in its own right.
target_link_libraries(
  consumer
  PRIVATE wwr::wwr.runtime_api
          wwr::wwr.blas
          wwr::wwr.wrappers.blas
          wwr::wwr.wrappers.solver
          wwr::wwr.wrappers.fft
          wwr::wwr.wrappers.sparse
          wwr::wwr.wrappers.tx
)
