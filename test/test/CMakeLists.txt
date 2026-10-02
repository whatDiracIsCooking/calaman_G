# test/test/ -- the GPU test suites. A suite links any utilities it needs from
# test/utils/ (added before this directory) and the shared modules under
# test/shared/.

add_subdirectory(columnwise_ell1)
add_subdirectory(columnwise_ell2)
add_subdirectory(complex_cast)
add_subdirectory(diff_norm)
add_subdirectory(expm)
add_subdirectory(feast)
add_subdirectory(gebal)
add_subdirectory(horner)
add_subdirectory(laset)
add_subdirectory(linalg)
add_subdirectory(orthogonalize)
add_subdirectory(paterson_stockmeyer)
add_subdirectory(random_unitary)
