# The warp-reduction example: a hand-written warp-level kernel on
# cooperative_groups.cuh, and the host code that runs it.
#
# This is the in-tree consumer that keeps src/cooperative_groups.cuh honest.
# test/gpu/cooperative_groups.cu compiles every portable entity the header
# reaches; nothing launched one until this, so the shfl_down ladder's answer was
# never checked on a device. Deliberately an example rather than a
# wwr.extension.* module: see example/README.md.
#
# Two targets because a .cu and a C++23 module TU cannot share one. The device
# source needs a language decision and, under HIP, `-x hip --offload-arch=...`;
# main.cpp is an ordinary host CXX compile that imports modules and must get
# neither. cmake/wwr_add_gpu_device_library.cmake is where that split is
# spelled out in full.

# wwr.device carries cooperative_groups.cuh and runtime.cuh (and the
# WWR_WARP_SIZE the tile size is built from); wwr.extension.bridge carries
# "extension/bridge/gpu_stream_bridge.h", which warp_reduce_bridge.h includes.
# PRIVATE, as every device-library link is: nothing that links this should
# inherit a device-code include path.
wwr_add_gpu_device_library(
  NAME
  wwr.example.warp_reduce.device
  SOURCES
  warp_reduce.cu
  LINK_PRIVATE
  wwr.device
  wwr.extension.bridge
)

# The host half. No ctest entry: it needs a real device, and the runtime tier
# currently has nothing device-dependent in it (see CLAUDE.md, "Running
# tests") -- registering this would turn every driverless box red. Run it by
# hand from the build directory.
add_executable(wwr.example.warp_reduce main.cpp)

# wwr.extension.bridge for gpu_stream_bridge.h, and wwr_backend for the
# WWR_GPU_BACKEND_* define that selected_backend.h behind it reads when
# there is no device-compile pass to supply __CUDACC__ / __HIP__ -- the same
# pair, for the same reason, that src/extension/rand's module links.
target_link_libraries(
  wwr.example.warp_reduce
  PRIVATE wwr.runtime_api
          wwr.extension.runtime
          wwr.extension.memory_buffer
          wwr.extension.bridge
          wwr_backend
          wwr.example.warp_reduce.device
)

set_target_properties(
  wwr.example.warp_reduce
  PROPERTIES CXX_MODULE_STD 1 CUDA_SEPARABLE_COMPILATION OFF
             CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)
