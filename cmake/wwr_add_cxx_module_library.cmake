# wwr_add_cxx_module_library — create a C++23 module library (the project's
# most-used target macro). See cmake/README.md, "WWR_ADD_CXX_MODULE_LIBRARY",
# for the full parameter list, defaults, and an example.

# ---------------------------------------------------------------------------
# Internal helpers (underscore-prefixed — not part of the public API)
# ---------------------------------------------------------------------------

# Add CUDAToolkit include directories with validated visibility.
function(_wwr_configure_cuda_toolkit target visibility)
  if(NOT visibility)
    set(visibility "PRIVATE")
  elseif(NOT visibility STREQUAL "PUBLIC" AND NOT visibility STREQUAL "PRIVATE")
    message(FATAL_ERROR "WWR_ADD_CXX_MODULE_LIBRARY:"
                        " INCLUDE_CUDA_TOOLKIT must be PUBLIC or PRIVATE"
    )
  endif()
  target_include_directories(
    ${target} SYSTEM ${visibility} ${CUDAToolkit_INCLUDE_DIRS}
  )
endfunction()

# ---------------------------------------------------------------------------
# Public macro
# ---------------------------------------------------------------------------
macro(WWR_ADD_CXX_MODULE_LIBRARY)
  set(_bool_opts "IMPORT_STD;NO_CUDA_DEVICE_LINKING")
  set(_single_opts "NAME;PRIMARY_INTERFACE;INCLUDE_CUDA_TOOLKIT")
  set(_multi_opts "PARTITIONS;IMPLEMENTATION;LINK_PUBLIC;LINK_PRIVATE"
                  ";INCLUDE_DIRS_PUBLIC;INCLUDE_DIRS_PRIVATE"
  )
  cmake_parse_arguments(
    ARG
    "${_bool_opts}"
    "${_single_opts}"
    "${_multi_opts}"
    ${ARGN}
  )

  _wwr_require_args(
    "WWR_ADD_CXX_MODULE_LIBRARY" ARG NAME PRIMARY_INTERFACE
  )

  add_library(${ARG_NAME})

  set(ALL_MODULE_INTERFACE_FILES ${ARG_PRIMARY_INTERFACE})
  if(ARG_PARTITIONS)
    list(APPEND ALL_MODULE_INTERFACE_FILES ${ARG_PARTITIONS})
  endif()

  target_sources(
    ${ARG_NAME} PUBLIC FILE_SET CXX_MODULES FILES ${ALL_MODULE_INTERFACE_FILES}
  )

  # PUBLIC compile feature, not the global CMAKE_CXX_STANDARD (a directory
  # variable, not exported): an installed consumer compiles these module units
  # in a synthetic target built from this target's INTERFACE properties, and
  # CMake refuses that target with "no C++ standard found" unless cxx_std_23
  # travels with it.
  target_compile_features(${ARG_NAME} PUBLIC cxx_std_23)

  if(ARG_IMPLEMENTATION)
    target_sources(${ARG_NAME} PRIVATE ${ARG_IMPLEMENTATION})
  endif()

  if(ARG_IMPORT_STD)
    set_target_properties(${ARG_NAME} PROPERTIES CXX_MODULE_STD 1)
  endif()

  if(ARG_NO_CUDA_DEVICE_LINKING)
    _wwr_disable_cuda_device_linking(${ARG_NAME})
  endif()

  if(DEFINED ARG_INCLUDE_CUDA_TOOLKIT)
    _wwr_configure_cuda_toolkit(${ARG_NAME} "${ARG_INCLUDE_CUDA_TOOLKIT}")
  endif()

  if(ARG_INCLUDE_DIRS_PUBLIC OR ARG_INCLUDE_DIRS_PRIVATE)
    _wwr_configure_include_dirs(
      ${ARG_NAME}
      PUBLIC
      ${ARG_INCLUDE_DIRS_PUBLIC}
      PRIVATE
      ${ARG_INCLUDE_DIRS_PRIVATE}
    )
  endif()

  if(ARG_LINK_PUBLIC OR ARG_LINK_PRIVATE)
    _wwr_configure_link_libraries(
      ${ARG_NAME}
      PUBLIC
      ${ARG_LINK_PUBLIC}
      PRIVATE
      ${ARG_LINK_PRIVATE}
    )
  endif()

  _wwr_create_alias(${ARG_NAME})

endmacro()
