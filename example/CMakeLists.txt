# The in-tree examples. See README.md for what belongs here.
#
# consumer/ is deliberately absent: it is a standalone project with its own
# project() call, configured against an INSTALLED gpumod by
# devtools/install-check.sh. Adding it here would give it gpumod's source tree,
# which is the one thing it must not have.
#
# Added unconditionally, WWR_COMPILE_TIME_ONLY included, so
# devtools/cross-backend-check.sh compiles what is here for the other backend.
# An example that only ever builds on one backend is not an example of portable
# code.
add_subdirectory(warp_reduce)

# The custom-default-error-policy example. Built here (both backends, so
# cross-backend-check covers it); run by hand -- see its CMakeLists.txt for why
# it has no ctest entry.
add_subdirectory(custom_default_error_policy)
