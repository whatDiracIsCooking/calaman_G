# calaman.reduce_columns -- backend-neutral, header-only per-column reduction.
# A device-code header (reduce_columns.cuh), not a C++23 module: consumers
# #include it directly into a .cu (CUDA) or -x hip device-compiled (HIP) TU, so
# it is an INTERFACE library carrying the header's include root and its device
# dependencies -- structurally wwr.extension.parallel_for, not a calaman module.
#
# It is NOT declared through calaman_add_interface_library: that macro roots the
# include path at this directory, which would spell the header
# "reduce_columns.cuh" from here only. A cross-directory consumer
# (src/columnwise_ell1/columnwise_ell1.cu) must reach it ROOT-RELATIVE as
# "reduce_columns/reduce_columns.cuh" (CLAUDE.md, #include style), so the root
# is ${CMAKE_SOURCE_DIR}/src -- the same hand-written shape WarpWraps'
# parallel_for uses for exactly this reason.

add_library(calaman.reduce_columns INTERFACE)
add_library(calaman::reduce_columns ALIAS calaman.reduce_columns)

# src/ itself is the include root: reduce_columns.cuh is reached as
# "reduce_columns/reduce_columns.cuh" and pulls in the src/-root "runtime.h"
# (via wwr.device) alongside it; "common/functor_concepts.h" (calaman.common's
# header) resolves off this same root, so no link on calaman.common is needed.
target_include_directories(
  calaman.reduce_columns
  INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:${CALAMAN_INSTALL_INCLUDEDIR}>
)

# wwr.device carries runtime.h's include path plus WWR_WARP_SIZE and the
# runtime-only libs (CUDA::cudart / hip::host, never hip::device). A
# device-KERNEL consumer of this header links hip::device itself, scoped to its
# own gpu_device_library -- see src/columnwise_ell1. runtime.h (wwrStream_t) and
# device_guard.h (device-pass gate) both resolve off that same src/ root, so
# wwr.device covers the wwr headers this one includes too.
target_link_libraries(
  calaman.reduce_columns INTERFACE wwr.device
)
