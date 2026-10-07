# calaman_add_kernel_name_check() — fail the build when a `__global__` entry in
# any calaman_add_gpu_device_library() archive does not mangle `calaman`.
#
# racecheck and synccheck are scoped with `--kernel-name kns=calaman`
# (docs/sanitizers.md, S2), so a kernel outside that namespace would be skipped
# by both tools without a word. This makes that a build error instead.
#
# CUDA only: it reads the cubins with cuobjdump, and the filter it protects
# exists only under compute-sanitizer. Call it after every add_subdirectory
# that can define a device library, since it reads their list back from the
# calaman_device_libraries umbrella.
function(calaman_add_kernel_name_check)
  if(NOT CALAMAN_GPU_BACKEND STREQUAL "CUDA")
    return()
  endif()
  find_program(
    CALAMAN_CUOBJDUMP_EXECUTABLE
    NAMES cuobjdump
    HINTS "${CUDAToolkit_BIN_DIR}" REQUIRED
  )

  get_property(
    _libs
    TARGET calaman_device_libraries
    PROPERTY MANUALLY_ADDED_DEPENDENCIES
  )
  set(_archives "")
  foreach(_lib IN LISTS _libs)
    list(APPEND _archives "$<TARGET_FILE:${_lib}>")
  endforeach()
  string(JOIN "|" _archive_arg ${_archives})

  set(_stamp "${CMAKE_BINARY_DIR}/calaman_kernel_names.stamp")
  add_custom_command(
    OUTPUT "${_stamp}"
    COMMAND
      ${CMAKE_COMMAND} "-DCUOBJDUMP=${CALAMAN_CUOBJDUMP_EXECUTABLE}"
      "-DARCHIVES=${_archive_arg}" "-DSTAMP=${_stamp}" -P
      ${CMAKE_SOURCE_DIR}/cmake/calaman_check_kernel_names.cmake
    DEPENDS ${_libs} ${_archives}
            ${CMAKE_SOURCE_DIR}/cmake/calaman_check_kernel_names.cmake
    COMMENT "Checking every calaman __global__ mangles 'calaman'"
    VERBATIM
  )
  add_custom_target(
    calaman_kernel_name_check ALL
    DEPENDS "${_stamp}"
    COMMENT "calaman kernel-name check"
  )
endfunction()
