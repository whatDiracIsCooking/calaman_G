# Spec suite for calaman.complex_cast -- real/complex array conversion. Like the
# gebal suite it needs NO reference LAPACK: the four operations are their own
# oracle (round-trip recovers the inputs, a set_* leaves the other component
# intact, count == 0 is a no-op), so there is no lapack_reference guard.
#
# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case stages
# the arrays on the device and runs the cast kernels. The two test/shared/
# modules back the device buffers that stage the inputs, while wwr.complex and
# wwr.wrappers.common supply the complex element types and the ComplexToRealType
# the real planes are spelled with.
calaman_add_gtest_executable(
  NAME
  complex_cast_tests
  SOURCES
  complex_cast_tests.cpp
  LINK
  calaman.complex_cast
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
  complex_cast_tests
  SUITES
  ComplexCastSpecTests
  REQUIRES_GPU
)
