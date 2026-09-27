# Guard for wwr_add_gtest_suite_tests(): fail if the SUITES/TYPED_SUITES
# a CMakeLists names have drifted from the suites the binary actually registers.
#
# Registering ctest entries per suite means the suite names live in CMake, by
# hand. Without a check, adding a test file and forgetting to list its suite
# loses that suite's coverage silently -- the build is green, ctest is green,
# and nothing ran. This repo has already been bitten by exactly that shape once
# (the FEAST tests passed vacuously for months), so the list gets verified
# rather than trusted.
#
# Run via:
#   cmake -DEXE=<test binary> -DEXPECTED=<a;b;c> \
#     -P wwr_check_gtest_suites.cmake

if(NOT EXE)
  message(FATAL_ERROR "EXE not set")
endif()

execute_process(
  COMMAND "${EXE}" --gtest_list_tests
  OUTPUT_VARIABLE _listing
  ERROR_VARIABLE _stderr
  RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
  message(
    FATAL_ERROR "'${EXE} --gtest_list_tests' failed (${_rc}):\n${_stderr}"
  )
endif()

# A suite header is a line starting in column 0 and ending in '.'; a typed suite
# carries the type index as a '/N' suffix, which is collapsed away so
# `FooTests/0.` and `FooTests/1.` both report as `FooTests`.
set(_actual "")
string(REPLACE "\n" ";" _lines "${_listing}")
foreach(_line IN LISTS _lines)
  if(_line MATCHES "^([A-Za-z_][A-Za-z0-9_]*)(/[0-9]+)?\\.")
    list(APPEND _actual "${CMAKE_MATCH_1}")
  endif()
endforeach()

list(REMOVE_DUPLICATES _actual)
list(SORT _actual)

set(_expected ${EXPECTED})
list(REMOVE_DUPLICATES _expected)
list(SORT _expected)

if(NOT _actual STREQUAL _expected)
  set(_missing ${_actual})
  list(REMOVE_ITEM _missing ${_expected})
  set(_stale ${_expected})
  list(REMOVE_ITEM _stale ${_actual})
  message(
    FATAL_ERROR
      "GoogleTest suite list is out of date for ${EXE}.\n"
      "  registered but NOT listed in CMake "
      "(these tests never run): ${_missing}\n"
      "  listed in CMake but not registered (stale entries): ${_stale}\n"
      "Update the SUITES/TYPED_SUITES argument in that target's CMakeLists.txt."
  )
endif()
