# WWR_INTERNAL_HELPERS.cmake Shared internal helpers used by multiple
# gpumod macros. Not part of the public API.

# Fail with a uniform "<label>: <arg> is required" message if any named argument
# is unset. `label` names the calling macro for the error, `prefix` is its
# cmake_parse_arguments output prefix, and the remaining arguments are the
# argument names to require. Reads the ${prefix}_<arg> variables out of the
# caller's scope (a function's variable reads fall through to it).
function(_wwr_require_args label prefix)
  foreach(_arg IN LISTS ARGN)
    if(NOT ${prefix}_${_arg})
      message(FATAL_ERROR "${label}: ${_arg} is required")
    endif()
  endforeach()
endfunction()

# Turn off CUDA device-side linking on a target. Separable compilation is OFF on
# every target in this tree, and this is the one-liner that states that
# invariant -- see cmake/README.md. A no-op under HIP, where CUDA is disabled.
function(_wwr_disable_cuda_device_linking target)
  set_target_properties(
    ${target} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
                         CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  )
endfunction()

# Create a :: alias for a dot-separated target name.
function(_wwr_create_alias target)
  string(FIND "${target}" "." _has_dot)
  if(_has_dot EQUAL -1)
    return()
  endif()
  string(REPLACE "." "::" _alias "${target}")
  add_library(${_alias} ALIAS ${target})
endfunction()

# Add explicit include directories (PUBLIC and/or PRIVATE).
function(_wwr_configure_include_dirs target)
  cmake_parse_arguments(
    _inc
    ""
    ""
    "PUBLIC;PRIVATE"
    ${ARGN}
  )
  set(args "")
  if(_inc_PUBLIC)
    list(APPEND args PUBLIC ${_inc_PUBLIC})
  endif()
  if(_inc_PRIVATE)
    list(APPEND args PRIVATE ${_inc_PRIVATE})
  endif()
  target_include_directories(${target} ${args})
endfunction()

# Link PUBLIC and/or PRIVATE dependencies.
function(_wwr_configure_link_libraries target)
  cmake_parse_arguments(
    _link
    ""
    ""
    "PUBLIC;PRIVATE"
    ${ARGN}
  )
  set(args "")
  if(_link_PUBLIC)
    list(APPEND args PUBLIC ${_link_PUBLIC})
  endif()
  if(_link_PRIVATE)
    list(APPEND args PRIVATE ${_link_PRIVATE})
  endif()
  target_link_libraries(${target} ${args})
endfunction()
