# wwr_add_test_executable — a plain ctest-registered executable (no
# GoogleTest), used by the compile-time tiers where the proof is that the TU
# compiled and linked. See cmake/README.md, "Tests", for when to use this versus
# the GoogleTest macros.
#
#   wwr_add_test_executable(
#     NAME           <target, also the ctest name>
#     [MAIN          <source with main(); default main.cpp>]
#     [TIMEOUT       <seconds; default 60>]
#     [LINK_PRIVATE  <lib ...>]
#     [WHOLE_ARCHIVE <target ...>]   # force objects in, for self-registering
#                                    # tests
#     [LINK_LIBSTDCXX])              # link libstdc++.so.6 explicitly
macro(wwr_add_test_executable)
  cmake_parse_arguments(
    _TEX
    "LINK_LIBSTDCXX" # Boolean options
    "NAME;MAIN;TIMEOUT" # Single-value arguments
    "LINK_PRIVATE;WHOLE_ARCHIVE" # Multi-value arguments
    ${ARGN}
  )

  _wwr_require_args("wwr_add_test_executable" _TEX NAME)

  if(NOT _TEX_MAIN)
    set(_TEX_MAIN "main.cpp")
  endif()

  if(NOT _TEX_TIMEOUT)
    set(_TEX_TIMEOUT 60)
  endif()

  add_executable(${_TEX_NAME} ${_TEX_MAIN})

  if(_TEX_LINK_PRIVATE)
    target_link_libraries(${_TEX_NAME} PRIVATE ${_TEX_LINK_PRIVATE})
  endif()

  foreach(_wa_target IN LISTS _TEX_WHOLE_ARCHIVE)
    target_link_options(
      ${_TEX_NAME}
      PRIVATE
      "-Wl,--whole-archive"
      "$<TARGET_FILE:${_wa_target}>"
      "-Wl,--no-whole-archive"
    )
  endforeach()

  if(_TEX_LINK_LIBSTDCXX)
    target_link_libraries(
      ${_TEX_NAME} PRIVATE /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    )
  endif()

  set_target_properties(${_TEX_NAME} PROPERTIES CXX_MODULE_STD 1)
  _wwr_disable_cuda_device_linking(${_TEX_NAME})

  add_test(NAME ${_TEX_NAME} COMMAND ${_TEX_NAME})
  set_tests_properties(${_TEX_NAME} PROPERTIES TIMEOUT ${_TEX_TIMEOUT})

endmacro()
