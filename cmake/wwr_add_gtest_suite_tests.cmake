# wwr_add_gtest_suite_tests — register one ctest entry per GoogleTest suite,
# instead of one per binary or one per case, plus a drift guard.
#
#   wwr_add_gtest_suite_tests(
#       TARGET       extension_math_tests
#       SUITES       PlainSuiteA PlainSuiteB      # TEST() / TEST_F() suites
#       TYPED_SUITES FooTests BarTests            # TYPED_TEST_SUITE() suites
#       TYPES        float double cuComplex cuDoubleComplex
#       TIMEOUT      120
#       REQUIRES_GPU)                             # label every entry `gpu`
#
# See cmake/README.md, "Tests", for why per suite (the cost measurement), how
# TYPES maps typed suites, and why a target may need more than one call.
#
# REQUIRES_GPU marks a binary that needs a live device -- it allocates device
# memory, launches a kernel, or creates a vendor-library handle. Nothing in
# test/ calls GTEST_SKIP and no case gates on a device count, so on a box with
# no card those suites ERROR rather than skip-and-name, and a red run there is
# indistinguishable from a broken change. The label is what lets a caller say
# which half it wants:
#
#   ctest --preset ci-cuda -LE gpu     everything that does not need a device
#   ctest --preset default             everything (the dev box, with a card)
#
# It is the same mechanism as the no_sanitizer label on the two allocation-
# failure suites, and deliberately so: an EXCLUDED test is named in the ctest
# output, where a skipped one blends into a green run. The ci-cuda preset that
# .github/workflows/ci.yml drives runs the excluding form, because a
# GitHub-hosted runner has no GPU.

# Internal: registers the drift guard once per target, after every
# wwr_add_gtest_suite_tests() call in the directory has contributed its
# suite names. Not meant to be called directly.
function(_wwr_register_gtest_suite_guard target)
  get_property(_suites GLOBAL PROPERTY _wwr_gtest_suites_${target})
  add_test(
    NAME ${target}.SuiteListIsComplete
    COMMAND
      ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:${target}> "-DEXPECTED=${_suites}"
      -P ${CMAKE_SOURCE_DIR}/cmake/wwr_check_gtest_suites.cmake
  )
  set_tests_properties(${target}.SuiteListIsComplete PROPERTIES TIMEOUT 60)

  # The guard is NOT labeled `gpu` even for a REQUIRES_GPU target. It runs the
  # binary with --gtest_list_tests, which enumerates the registry and returns
  # without constructing a fixture, so it needs the binary to LOAD but never
  # touches a device -- and it is exactly the check that keeps the hand-written
  # suite list honest, which is worth having on a GPU-less runner too. If a
  # target ever grows a static initializer that talks to the driver, this is
  # where it would start failing on such a box, and the fix is to label the
  # guard rather than to delete it.
endfunction()

# Register a GoogleTest binary with ctest as one entry per suite, plus the
# <target>.SuiteListIsComplete drift guard. TYPED_SUITES x TYPES expands to one
# entry per instantiation. See cmake/README.md, "Tests", for the why.
function(wwr_add_gtest_suite_tests)
  cmake_parse_arguments(
    _GST
    "REQUIRES_GPU"
    "TARGET;TIMEOUT"
    "SUITES;TYPED_SUITES;TYPES"
    ${ARGN}
  )

  _wwr_require_args("wwr_add_gtest_suite_tests" _GST TARGET)
  if(NOT _GST_TIMEOUT)
    set(_GST_TIMEOUT 120)
  endif()
  if(_GST_TYPED_SUITES AND NOT _GST_TYPES)
    message(
      FATAL_ERROR
        "wwr_add_gtest_suite_tests: TYPED_SUITES given without TYPES"
    )
  endif()

  # Collected rather than applied inline so a plain and a typed suite cannot
  # drift apart, and so a future second label has one place to go.
  set(_labels "")
  if(_GST_REQUIRES_GPU)
    list(APPEND _labels gpu)
  endif()

  # Plain suites: one entry, filter `Suite.*`.
  foreach(_suite IN LISTS _GST_SUITES)
    add_test(NAME ${_suite} COMMAND ${_GST_TARGET} --gtest_filter=${_suite}.*)
    set_tests_properties(${_suite} PROPERTIES TIMEOUT ${_GST_TIMEOUT})
    if(_labels)
      set_property(TEST ${_suite} APPEND PROPERTY LABELS ${_labels})
    endif()
  endforeach()

  # Typed suites: one entry per (suite, type), filter `Suite/<index>.*`.
  list(LENGTH _GST_TYPES _type_count)
  foreach(_suite IN LISTS _GST_TYPED_SUITES)
    set(_index 0)
    while(_index LESS _type_count)
      list(GET _GST_TYPES ${_index} _type_name)
      add_test(NAME "${_suite}<${_type_name}>"
               COMMAND ${_GST_TARGET} --gtest_filter=${_suite}/${_index}.*
      )
      set_tests_properties(
        "${_suite}<${_type_name}>" PROPERTIES TIMEOUT ${_GST_TIMEOUT}
      )
      if(_labels)
        set_property(TEST "${_suite}<${_type_name}>" APPEND PROPERTY LABELS
                                                                    ${_labels}
        )
      endif()
      math(EXPR _index "${_index} + 1")
    endwhile()
  endforeach()

  # Drift guard: suite names accumulate on a global property and the guard is
  # deferred to end-of-directory-scope, so a target split across several calls
  # is checked against the union (cmake/README.md, "Tests").
  set_property(
    GLOBAL APPEND PROPERTY _wwr_gtest_suites_${_GST_TARGET} ${_GST_SUITES}
                           ${_GST_TYPED_SUITES}
  )

  get_property(
    _guard_scheduled GLOBAL PROPERTY _wwr_gtest_guard_${_GST_TARGET}
  )
  if(NOT _guard_scheduled)
    set_property(GLOBAL PROPERTY _wwr_gtest_guard_${_GST_TARGET} TRUE)
    # EVAL CODE bakes the target name into the deferred call as a literal.
    # A plain `DEFER CALL f("${_GST_TARGET}")` does not work: deferred
    # arguments are re-evaluated when the call finally runs, by which point
    # this function's scope is gone and the argument expands to nothing --
    # which surfaces as `$<TARGET_FILE:>` failing to parse.
    set(_guard_fn _wwr_register_gtest_suite_guard)
    cmake_language(
      EVAL CODE "cmake_language(DEFER CALL ${_guard_fn} \"${_GST_TARGET}\")"
    )
  endif()
endfunction()
