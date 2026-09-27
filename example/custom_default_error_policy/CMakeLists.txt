# The custom-default-error-policy example.
#
# Demonstrates the WWR_DEFAULT_ERROR_POLICY_IMPL build option: configure the
# tree with
#   -DWWR_DEFAULT_ERROR_POLICY_IMPL=\
#       ${CMAKE_CURRENT_SOURCE_DIR}/custom_default_error_policy.h
# and wwr::extension::DefaultErrorPolicy is replaced build-wide (see
# custom_default_error_policy.h and src/extension/common/error_handling). The
# swap happens in that module's own build; this target just imports the
# extension layer and inherits whichever default policy the tree was compiled
# with, so it builds unchanged either way.
#
# No ctest entry -- and this is why: with the default policy the program aborts
# by design (it triggers an error to show the policy running), so registering it
# would turn a default build's test run red. It needs no device; run it by hand,
# both ways, to see the difference (see example/README.md).
add_executable(wwr.example.custom_default_error_policy main.cpp)

# wwr.runtime_api for gpuErrorInvalidValue/gpuSuccess; wwr.extension.common for
# gpu_check and DefaultErrorPolicy. Both are backend-neutral, so this compiles
# on either backend -- which is what devtools/cross-backend-check.sh checks.
target_link_libraries(
  wwr.example.custom_default_error_policy
  PRIVATE wwr.runtime_api
          wwr.extension.common
)

set_target_properties(
  wwr.example.custom_default_error_policy PROPERTIES CXX_MODULE_STD 1
)
