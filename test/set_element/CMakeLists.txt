# Spec suite for calaman.set_element -- read one element of a device array by a
# device-held index. Like the complex_cast suite it needs NO reference LAPACK:
# the spec is its own oracle (a gather copies the element, a magnitude is the
# host modulus, and iamax's device index resolves to the host's max |element|),
# so there is no lapack_reference guard.
#
# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case stages
# the arrays on the device and runs the kernel. The two test/shared/ modules
# back the device buffers that stage the inputs; wwr.blas / wwr.wrappers.blas
# drive iamax in device pointer mode for the integration case; wwr.complex and
# wwr.wrappers.common supply the complex element types and the ComplexToRealType
# the magnitude output is spelled with.
calaman_add_gtest_executable(
  NAME
  set_element_tests
  SOURCES
  set_element_tests.cpp
  LINK
  calaman::set_element
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  wwr::blas
  wwr::wrappers::blas
  wwr::runtime_api
  wwr::complex
  wwr::wrappers::common
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  set_element_tests
  SUITES
  SetElementSpecTests
  REQUIRES_GPU
)
