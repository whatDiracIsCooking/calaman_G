# Spec suite for calaman.gebal -- matrix balancing. Unlike the linalg oracle
# suites it needs NO reference LAPACK: the specification is its own oracle (undo
# the reported permutation and scaling and the input returns bit for bit, a
# closed-form 2x2, a diagonal matrix, n == 1), so there is no
# calaman::lapack_reference guard here.
#
# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case stages
# A/scale/work on the device and runs the mark/pick/swap/sweep kernels. The two
# test/shared/ modules back the device buffers that stage the inputs, while
# wwr.complex and wwr.wrappers.common supply the complex element types and the
# ComplexToRealType the real scale array is spelled with.
calaman_add_gtest_executable(
  NAME
  gebal_tests
  SOURCES
  gebal_tests.cpp
  LINK
  calaman.gebal
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.utils.shared_device
  wwr.runtime_api
  wwr.complex
  wwr.wrappers.common
  wwr.extension.memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  gebal_tests
  SUITES
  GebalSpecTests
  REQUIRES_GPU
)
