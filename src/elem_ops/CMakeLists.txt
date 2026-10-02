# calaman.elem_ops -- header-only, backend-neutral per-element arithmetic.
# A device-code header (elem_ops.cuh), not a C++23 module: consumers #include it
# into a .cu (CUDA) or -x hip device-compiled (HIP) TU, so it is an INTERFACE
# library carrying the header's include root and its device dependency
# (wwr.device) -- structurally the same as calaman.reduce_columns, not a calaman
# module. It is device-only because the complex arithmetic it wraps exists only
# in a device pass, so it has no module face and nothing host-side imports it.
#
# NOT declared through calaman_add_interface_library: that macro roots the
# include path at THIS directory, which would spell the header "elem_ops.cuh"
# from here only. A cross-directory consumer (src/expm/expm.cu, ...) reaches it
# ROOT-RELATIVE as "elem_ops/elem_ops.cuh" (CLAUDE.md, #include style), so the
# root is ${CMAKE_SOURCE_DIR}/src -- the same hand-written shape reduce_columns
# and WarpWraps' parallel_for use for exactly this reason.

add_library(calaman.elem_ops INTERFACE)
add_library(calaman::elem_ops ALIAS calaman.elem_ops)

# src/ itself is the include root: elem_ops.cuh is reached as
# "elem_ops/elem_ops.cuh" and pulls in the WarpWraps-root "complex.h" and
# "wrappers/math/math.cuh" (both via wwr.device) alongside it.
target_include_directories(
  calaman.elem_ops
  INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:${CALAMAN_INSTALL_INCLUDEDIR}>
)

# wwr.device carries the WarpWraps src/ include root (where complex.h and
# wrappers/math/math.cuh resolve) plus the runtime-only libs. A device-KERNEL
# consumer of this header links hip::device itself, scoped to its own
# gpu_device_library -- see src/expm. complex.h's arithmetic and math.cuh's
# intrinsics are __device__ __forceinline__, so there is no separate link beyond
# the runtime wwr.device already provides.
target_link_libraries(
  calaman.elem_ops INTERFACE wwr.device
)
