# calaman_add_sanitizer_canary — register a deliberately buggy run as a ctest
# entry that passes only when the sanitizer catches the bug. A canary that goes
# red means the check it guards went dark (docs/sanitizers.md).
#
#   calaman_add_sanitizer_canary(
#     NAME     <ctest name>
#     TARGET   <executable target>
#     ARGS     <arg ...>                 # selects the bug to run
#     EXPECT   <regex>                   # the sanitizer's report line
#     [LAUNCHER <cmd ...>]               # e.g. ${CMAKE_TEST_LAUNCHER}
#     [REQUIRES_GPU])                    # label `gpu` too
#
# Not WILL_FAIL: that credits any non-zero exit, so a crash for the wrong
# reason passes, and a report printed with exit 0 (the tool not failing the
# run) is invisible. calaman_check_sanitizer_canary.cmake demands both a
# non-zero exit AND the EXPECT report. Every entry is labeled
# `sanitizer_canary`; the caller registers it only under its own preset.
function(calaman_add_sanitizer_canary)
  cmake_parse_arguments(
    _CAN
    "REQUIRES_GPU"
    "NAME;TARGET;EXPECT"
    "ARGS;LAUNCHER"
    ${ARGN}
  )
  _calaman_require_args(
    "calaman_add_sanitizer_canary"
    _CAN
    NAME
    TARGET
    EXPECT
  )

  # A ';' would split the -D argument; the script turns '|' back into ';'.
  string(JOIN "|" _launcher ${_CAN_LAUNCHER})
  string(JOIN "|" _args ${_CAN_ARGS})

  add_test(
    NAME ${_CAN_NAME}
    COMMAND
      ${CMAKE_COMMAND} "-DLAUNCHER=${_launcher}"
      -DEXE=$<TARGET_FILE:${_CAN_TARGET}> "-DARGS=${_args}"
      "-DEXPECT=${_CAN_EXPECT}" -P
      ${CMAKE_SOURCE_DIR}/cmake/calaman_check_sanitizer_canary.cmake
  )

  set(_labels sanitizer_canary)
  if(_CAN_REQUIRES_GPU)
    list(APPEND _labels gpu)
  endif()
  math(EXPR _timeout "60 * ${CALAMAN_TEST_TIMEOUT_MULTIPLIER}")
  set_tests_properties(
    ${_CAN_NAME} PROPERTIES LABELS "${_labels}" TIMEOUT ${_timeout}
  )
endfunction()
