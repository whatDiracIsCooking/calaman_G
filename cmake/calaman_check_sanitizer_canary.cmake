# Verdict for calaman_add_sanitizer_canary(): run the canary and pass only when
# it exited non-zero AND its output carries the sanitizer report EXPECT names.
#
# Run via:
#   cmake -DLAUNCHER=<a|b|c> -DEXE=<binary> -DARGS=<x|y> -DEXPECT=<regex> \
#     -P calaman_check_sanitizer_canary.cmake

if(NOT EXE OR NOT EXPECT)
  message(FATAL_ERROR "EXE and EXPECT must be set")
endif()
string(REPLACE "|" ";" _launcher "${LAUNCHER}")
string(REPLACE "|" ";" _args "${ARGS}")

execute_process(
  COMMAND ${_launcher} "${EXE}" ${_args}
  OUTPUT_VARIABLE _out
  ERROR_VARIABLE _out
  RESULT_VARIABLE _rc
)
message("${_out}")

if(_rc EQUAL 0)
  message(
    FATAL_ERROR
      "Canary exited 0: the sanitizer did not fail the run, so the check it "
      "guards is off or is not failing runs (docs/sanitizers.md)."
  )
endif()
if(NOT _out MATCHES "${EXPECT}")
  message(
    FATAL_ERROR
      "Canary exited ${_rc} but without the expected report '${EXPECT}': it "
      "failed for some other reason, which proves nothing about the check."
  )
endif()
message(STATUS "Canary tripped as expected (exit ${_rc}): ${EXPECT}")
