# test/test/ -- the GPU test suites. Each links the utilities it needs from
# test/utils/ (added before this directory, so those targets already exist).

add_subdirectory(elementwise_compare)
add_subdirectory(lacpy)
