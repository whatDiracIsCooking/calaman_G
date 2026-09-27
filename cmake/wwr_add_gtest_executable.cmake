# wwr_add_gtest_executable.cmake
# Provides wwr_add_gtest_executable macro for GoogleTest executables.
#
# Usage: wwr_add_gtest_executable(
#   NAME executable_name
#   SOURCES src1.cpp src2.cpp ...
#   [LINK dep1 dep2 ...]
# )
#
# Parameters:
#   NAME     - Executable target name; also used as the CTest target
#   SOURCES  - This binary's own test source files. Omit main.cpp --
#              GTest::gtest_main supplies main().
#   LINK     - Module libraries this binary's own SOURCES need
#
# Consolidates what every GoogleTest executable under test/ repeats: the
# CXX_MODULE_STD / CXX_SCAN_FOR_MODULES / CUDA_SEPARABLE_COMPILATION /
# CUDA_RESOLVE_DEVICE_SYMBOLS target properties and the GTest::gtest_main link.
macro(wwr_add_gtest_executable)
  cmake_parse_arguments(
    _GTE
    ""
    "NAME"
    "SOURCES;LINK"
    ${ARGN}
  )

  _wwr_require_args("wwr_add_gtest_executable" _GTE NAME SOURCES)

  add_executable(${_GTE_NAME} ${_GTE_SOURCES})
  target_link_libraries(${_GTE_NAME} PRIVATE GTest::gtest_main ${_GTE_LINK})

  set_target_properties(
    ${_GTE_NAME} PROPERTIES CXX_MODULE_STD 1 CXX_SCAN_FOR_MODULES ON
  )
  _wwr_disable_cuda_device_linking(${_GTE_NAME})
endmacro()
