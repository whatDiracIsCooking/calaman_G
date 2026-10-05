# Oracle suite for calaman.lascl2 (built under src/lascl2/) -- diagonal
# row-scaling of a matrix, X <-- D * X. There is no LAPACKE (or CBLAS) binding
# for ?lascl2, so unlike the laset suite this one needs no reference LAPACK: the
# oracle is a same-precision host transcription of ?lascl2's double loop, in
# the test source itself. Nothing to guard on, so no early return.
#
# REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`): every case stages
# the matrix and the diagonal vector on the device and runs the scaling kernel.
# wwr.extension.memory_buffer and the two test/shared/ modules back the device
# buffers that stage inputs; lascl2 itself takes a bare wwrStream_t and
# allocates nothing.
calaman_add_gtest_executable(
  NAME
  lascl2_tests
  SOURCES
  lascl2_tests.cpp
  LINK
  calaman::lascl2
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::utils::shared_device
  wwr::runtime_api
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  lascl2_tests
  SUITES
  Lascl2OracleTests
  REQUIRES_GPU
)
