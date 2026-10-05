# Oracle test for calaman.geev -- the ?geev driver: eigenvalues and optional
# left/right eigenvectors of a general real matrix, chaining the whole
# nonsymmetric eigenproblem (gebal -> gehrd -> orghr/hseqr -> trevc3 ->
# back-transform -> gebak -> normalize). REQUIRES_GPU: every case stages A on
# the device and runs the full kernel chain, so `ctest -LE gpu` excludes it. It
# compares with the reference LAPACK ?geev, which arrives as
# calaman::lapack_reference; guard on that target and return early when it is
# absent -- a missing oracle is a missing TIER named in the configure output,
# not a silent pass (docs/architecture.md section 3). ?geev HAS a LAPACKE C
# binding, so the suite uses LAPACKE_sgeev/dgeev (via lapacke.h). The two shared
# modules and wwr.extension.memory_buffer stage the device data; geev needs a
# BLAS handle (gehrd, the gemm back-transform, the normalization) and a solver
# handle (orghr), so the suite links wwr.blas and wwr.solver too.
if(NOT TARGET calaman::lapack_reference)
  message(STATUS "calaman::lapack_reference not found; skipping geev tests")
  return()
endif()

calaman_add_gtest_executable(
  NAME
  geev_tests
  SOURCES
  geev_tests.cpp
  LINK
  calaman::geev
  calaman::test::shared::abort_policy
  calaman::test::shared::device_handle
  calaman::test::shared::tolerance
  calaman::test::utils::shared_device
  calaman::lapack_reference
  wwr::runtime_api
  wwr::blas
  wwr::solver
  wwr::extension::memory_buffer
)

calaman_add_gtest_suite_tests(
  TARGET
  geev_tests
  SUITES
  GeevOracleTests
  REQUIRES_GPU
)
