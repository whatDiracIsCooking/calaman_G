# WWR_ADD_INTERFACE_LIBRARY.cmake Provides
# WWR_ADD_INTERFACE_LIBRARY macro for creating INTERFACE libraries that
# expose header files (.h/.cuh) to consumers

# ---------------------------------------------------------------------------
# Public macro
# ---------------------------------------------------------------------------

# Macro to create an INTERFACE library with standard boilerplate. Usage:
# WWR_ADD_INTERFACE_LIBRARY( NAME library_name )
#
# Parameters: NAME - Target name (e.g. wwr.core.parallel_for)
macro(WWR_ADD_INTERFACE_LIBRARY)
  set(_single_opts "NAME")
  set(_multi_opts "LINK_PUBLIC")
  cmake_parse_arguments(
    ARG
    ""
    "${_single_opts}"
    "${_multi_opts}"
    ${ARGN}
  )

  _wwr_require_args("WWR_ADD_INTERFACE_LIBRARY" ARG NAME)

  add_library(${ARG_NAME} INTERFACE)

  target_include_directories(
    ${ARG_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                          $<INSTALL_INTERFACE:${WWR_INSTALL_INCLUDEDIR}>
  )

  if(ARG_LINK_PUBLIC)
    target_link_libraries(${ARG_NAME} INTERFACE ${ARG_LINK_PUBLIC})
  endif()

  _wwr_create_alias(${ARG_NAME})

endmacro()
