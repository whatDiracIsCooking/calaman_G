# Sanitizer canaries: deliberately buggy runs that must FAIL under the check
# they guard, registered through calaman_add_sanitizer_canary() so that a check
# going dark turns its canary red. Each one is configured only under its own
# preset and is absent everywhere else; docs/sanitizers.md lists which canary
# proves which entry.
#
#   asan / hip-asan      host: heap-buffer-overflow in a .cpp and in a .cu's
#                        host pass (ASan), leak (LSan)
#   ubsan / hip-ubsan    host: signed overflow in a .cpp and in a .cu's host
#                        pass (UBSan)
#   compute-sanitizer    device: the canary for CALAMAN_COMPUTE_SANITIZER_TOOL

if(CALAMAN_ENABLE_ASAN OR CALAMAN_ENABLE_UBSAN)
  # Plain add_executable, not calaman_add_test_executable: that macro also
  # registers a bare run of the binary, which is not a canary.
  add_executable(sanitizer_canary_host host_main.cpp)

  # Host-side defects inside a .cu, in a device library like any module's so
  # it gets the flags their host passes get. Its own binary, so the one above
  # stays free of the GPU runtime.
  calaman_add_gpu_device_library(
    NAME
    calaman.sanitizer_canaries.host_cu.device
    SOURCES
    host_cu_canary.cu
    LINK_PRIVATE
    wwr::device
  )
  add_executable(sanitizer_canary_host_cu host_cu_main.cpp)
  target_link_libraries(
    sanitizer_canary_host_cu PRIVATE calaman.sanitizer_canaries.host_cu.device
  )
  _calaman_disable_cuda_device_linking(sanitizer_canary_host_cu)
endif()

if(CALAMAN_ENABLE_ASAN)
  calaman_add_sanitizer_canary(
    NAME
    sanitizer_canary.asan.heap_buffer_overflow
    TARGET
    sanitizer_canary_host
    ARGS
    heap-buffer-overflow
    EXPECT
    "ERROR: AddressSanitizer: heap-buffer-overflow"
  )
  calaman_add_sanitizer_canary(
    NAME
    sanitizer_canary.lsan.leak
    TARGET
    sanitizer_canary_host
    ARGS
    leak
    EXPECT
    "ERROR: LeakSanitizer: detected memory leaks"
  )
  calaman_add_sanitizer_canary(
    NAME
    sanitizer_canary.asan.cu_heap_buffer_overflow
    TARGET
    sanitizer_canary_host_cu
    ARGS
    heap-buffer-overflow
    EXPECT
    "ERROR: AddressSanitizer: heap-buffer-overflow"
  )
endif()

if(CALAMAN_ENABLE_UBSAN)
  calaman_add_sanitizer_canary(
    NAME
    sanitizer_canary.ubsan.signed_overflow
    TARGET
    sanitizer_canary_host
    ARGS
    signed-overflow
    EXPECT
    "host_main.cpp:[0-9:]+ runtime error: signed integer overflow"
  )
  calaman_add_sanitizer_canary(
    NAME
    sanitizer_canary.ubsan.cu_signed_overflow
    TARGET
    sanitizer_canary_host_cu
    ARGS
    signed-overflow
    EXPECT
    "host_cu_canary.cu:[0-9:]+ runtime error: signed integer overflow"
  )
endif()

if(CALAMAN_COMPUTE_SANITIZER)
  # A device library like any module's, so the build-time kernel-name check
  # (calaman_add_kernel_name_check) covers these kernels too.
  calaman_add_gpu_device_library(
    NAME
    calaman.sanitizer_canaries.device
    SOURCES
    device_canaries.cu
    LINK_PRIVATE
    wwr::device
  )
  add_executable(sanitizer_canary_device device_main.cpp)
  target_link_libraries(
    sanitizer_canary_device PRIVATE calaman.sanitizer_canaries.device
  )
  _calaman_disable_cuda_device_linking(sanitizer_canary_device)

  set(_tool "${CALAMAN_COMPUTE_SANITIZER_TOOL}")
  if(_tool STREQUAL "memcheck")
    # The leak needs --leak-check full on the launcher (docs/sanitizers.md).
    set(_cases "memcheck" "Invalid __global__ write" "memcheck-leak"
               "Leaked [0-9]+ bytes"
    )
  elseif(_tool STREQUAL "initcheck")
    set(_cases "initcheck" "Uninitialized __global__ memory read")
  elseif(_tool STREQUAL "racecheck")
    # Both severities: a WARNING-graded hazard must fail the run too (G6).
    set(_cases "racecheck-error" "Error: Race reported" "racecheck-warning"
               "Warning: Race reported"
    )
  elseif(_tool STREQUAL "synccheck")
    set(_cases "synccheck" "Barrier error")
  else()
    message(FATAL_ERROR "No sanitizer canary for tool '${_tool}'")
  endif()

  while(_cases)
    list(POP_FRONT _cases _case _expect)
    string(REPLACE "-" "_" _id "${_case}")
    calaman_add_sanitizer_canary(
      NAME
      sanitizer_canary.compute_sanitizer.${_id}
      TARGET
      sanitizer_canary_device
      ARGS
      ${_case}
      EXPECT
      "${_expect}"
      LAUNCHER
      ${CMAKE_TEST_LAUNCHER}
      REQUIRES_GPU
    )
  endwhile()
endif()
