# Tests for the experimental/ tier. Added from test/CMakeLists.txt only under
# CALAMAN_BUILD_EXPERIMENTAL (OFF by default), mirroring the src-side guard. One
# directory per module, same shape as the rest of test/.
add_subdirectory(xor_delta)
