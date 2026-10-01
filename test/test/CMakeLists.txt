# test/test/ -- the GPU test suites. A suite links any utilities it needs from
# test/utils/ (added before this directory) and the shared modules under
# test/shared/.

add_subdirectory(diff_norm)
add_subdirectory(linalg)
