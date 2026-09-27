# wwr_add_dispatch_check
#
# Check, at build time, that every explicit instantiation of an extension
# module's wrappers calls exactly the vendor function a hand-written TOML table
# names.
#
# Not a static_assert: C++ cannot see which function a wrapper body calls, so
# test/shared/dispatch.py disassembles the module's compiled objects instead and
# maps the relocations back through the WWR_FUNCTION table in the matching
# gpu* module. It catches what the type system lets through -- an int widened
# into a _64 entry point, iamax dispatching to amin, potrf to potri. Like
# WWR_LINK_CHECK, a failure fails the build. See the script's docstring for
# the details, and each table's header for the module's own naming rule.
#
#   wwr_add_dispatch_check(
#     NAME       wwr.test.wrappers.blas_dispatch  # the custom target
#     TARGET     wwr.wrappers.blas                # whose objects to read
#     TABLE      blas_dispatch.toml                   # relative to this dir
#     GPU_SOURCE src/blas.cppm                    # from PROJECT_SOURCE_DIR
#   )
#
# The new target is added to `compile_time_tests`, so it is built by the
# compile-time tier as well as by ALL.

# llvm-objdump and llvm-cxxfilt come from the same LLVM install as the compiler
# rather than from PATH: a system llvm-objdump of a different major version
# demangles Itanium module names differently, and the wrapper regex stops
# matching. Resolved on first use and cached, so a build that adds no dispatch
# check never needs them (they are in DOCTOR_OPTIONAL_TOOLS for that reason).
function(_wwr_find_dispatch_tools)
  get_filename_component(_cxx_real "${CMAKE_CXX_COMPILER}" REALPATH)
  get_filename_component(_llvm_bin "${_cxx_real}" DIRECTORY)
  find_program(
    WWR_LLVM_OBJDUMP
    NAMES llvm-objdump
    HINTS "${_llvm_bin}" REQUIRED
  )
  find_program(
    WWR_LLVM_CXXFILT
    NAMES llvm-cxxfilt
    HINTS "${_llvm_bin}" REQUIRED
  )
  # GLOBAL, because this runs inside a function: without it both the
  # Python3::Interpreter imported target and Python3_EXECUTABLE are scoped to
  # this call and are gone by the time add_custom_command expands them -- which
  # silently yields a COMMAND that tries to exec the .py file itself, and a
  # build that dies with "Permission denied".
  find_package(Python3 REQUIRED COMPONENTS Interpreter GLOBAL)
endfunction()

#!
# Add a build-time dispatch check over TARGET's compiled objects.
#
# :param NAME: name of the custom target to create
# :param TARGET: the module library whose objects are disassembled
# :param TABLE: the TOML dispatch table, relative to the current source dir
# :param GPU_SOURCE: the gpu* module holding WWR_FUNCTION, from the root
function(wwr_add_dispatch_check)
  cmake_parse_arguments(
    ARG
    ""
    "NAME;TARGET;TABLE;GPU_SOURCE"
    ""
    ${ARGN}
  )
  _wwr_require_args(
    "wwr_add_dispatch_check"
    ARG
    NAME
    TARGET
    TABLE
    GPU_SOURCE
  )
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "wwr_add_dispatch_check: unexpected arguments: "
                        "${ARG_UNPARSED_ARGUMENTS}"
    )
  endif()

  _wwr_find_dispatch_tools()

  set(_script "${PROJECT_SOURCE_DIR}/test/shared/dispatch.py")
  set(_table "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_TABLE}")
  set(_gpu_source "${PROJECT_SOURCE_DIR}/${ARG_GPU_SOURCE}")
  set(_stamp "${CMAKE_CURRENT_BINARY_DIR}/${ARG_NAME}.stamp")

  add_custom_command(
    OUTPUT "${_stamp}"
    COMMAND
      Python3::Interpreter "${_script}" --objdump "${WWR_LLVM_OBJDUMP}"
      --cxxfilt "${WWR_LLVM_CXXFILT}" --backend "${WWR_GPU_BACKEND}"
      --table "${_table}" --gpu-source "${_gpu_source}" --stamp "${_stamp}"
      "$<TARGET_OBJECTS:${ARG_TARGET}>"
    DEPENDS ${ARG_TARGET} "${_script}" "${_table}" "${_gpu_source}"
    COMMAND_EXPAND_LISTS VERBATIM
    COMMENT "Checking ${ARG_TARGET} dispatch"
  )
  add_custom_target(
    ${ARG_NAME} ALL
    DEPENDS "${_stamp}"
    COMMENT "${ARG_TARGET} dispatch check"
  )
  add_dependencies(compile_time_tests ${ARG_NAME})
endfunction()
