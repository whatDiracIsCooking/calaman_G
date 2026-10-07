# Sanitizer canaries: deliberately buggy runs that must FAIL under the check
# they guard, registered through calaman_add_sanitizer_canary() so that a check
# going dark turns its canary red. Each one is configured only under its own
# preset and is absent everywhere else; docs/sanitizers.md lists which canary
# proves which entry.
#
#   asan / hip-asan      host: heap-buffer-overflow (ASan), leak (LSan)
#   compute-sanitizer    device: the canary for CALAMAN_COMPUTE_SANITIZER_TOOL

if(CALAMAN_ENABLE_ASAN)
  # Plain add_executable, not calaman_add_test_executable: that macro also
  # registers a bare run of the binary, which is not a canary.
  add_executable(sanitizer_canary_host host_main.cpp)

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
    set(_cases "memcheck" "Invalid __global__ write")
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
