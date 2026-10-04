# Suite for calaman.davidson. The module is a skeleton: :buffer_size is complete
# (workspace sizing + carving), :solve is a NOT_SUPPORTED stub, so there is no
# numerical oracle here yet and no calaman::lapack_reference guard. The two
# test/shared/ modules back the device buffer the carve test lays the workspace
# into; wwr.solver supplies the handle davidson_bufferSize queries syevd with.
calaman_add_gtest_executable(
  NAME
  davidson_tests
  SOURCES
  davidson_tests.cpp
  LINK
  calaman.davidson
  calaman.test.shared.abort_policy
  calaman.test.shared.device_handle
  calaman.test.utils.shared_device
  wwr.blas
  wwr.solver
  wwr.runtime_api
  wwr.wrappers.common
  wwr.extension.memory_buffer
)

# The argument-checking contract is a pure host call -- davidson_bufferSize /
# make_davidson_slices reject a bad shape (or a null out-pointer) before using
# the handle -- so it runs on a card-less runner.
calaman_add_gtest_suite_tests(
  TARGET
  davidson_tests
  SUITES
  DavidsonArgCheckTests
)

# The sizing/carve suite creates a cuSOLVER handle (to query the eigensolver
# workspace) and allocates the workspace on the device, so it is REQUIRES_GPU
# (`gpu`, excluded by ctest -LE gpu).
calaman_add_gtest_suite_tests(
  TARGET
  davidson_tests
  SUITES
  DavidsonBufferSizeTests
  REQUIRES_GPU
)
