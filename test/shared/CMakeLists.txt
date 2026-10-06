# test/shared/ -- test-support modules imported by more than one test target.
# All pure host module interfaces: no device code, no GoogleTest, so they carry
# neither a .cu nor an explicit-instantiation unit. Not shipped, and nothing
# under src/ may link them -- see this directory's README.md for why these
# types live here rather than in src/.

# The abort-on-failure error policy the suites name explicitly, now that
# WarpWraps ships no default. Every test that instantiates a wrapper or calls
# gpu_check gets its policy from here. wwr.extension.common carries
# success_code / error_name / error_string (and re-exports
# wwr.extension.error_handling, hence gpu_check).
calaman_add_cxx_module_library(
  NAME
  calaman.test.shared.abort_policy
  PRIMARY_INTERFACE
  abort_policy.cppm
  LINK_PUBLIC
  wwr::extension::common
  IMPORT_STD
)

# The concrete device_handle the suites allocate DeviceBufferWrapper against.
# wwr.extension.handle carries the device_handle ladder this satisfies,
# wwr.extension.runtime the StreamWrapper / MemPoolWrapper it owns, and
# wwr.runtime_api wwrDeviceProp / wwrGetDeviceProperties.
calaman_add_cxx_module_library(
  NAME
  calaman.test.shared.device_handle
  PRIMARY_INTERFACE
  device_handle.cppm
  LINK_PUBLIC
  wwr::runtime_api
  wwr::extension::common
  wwr::extension::handle
  wwr::extension::runtime
  calaman::test::shared::abort_policy
  IMPORT_STD
)

# The shared factorization tolerance O(eps * ||A|| * min(m,n)) the linalg oracle
# suites compare against, plus eps<T>() and a host Frobenius norm. Pure host
# math on std:: only, so it links nothing -- just needs IMPORT_STD.
calaman_add_cxx_module_library(
  NAME
  calaman.test.shared.tolerance
  PRIMARY_INTERFACE
  tolerance.cppm
  IMPORT_STD
)

# EXPECT_CONVERGED, a header (a macro cannot ride an import) included as
# "shared/expect_converged.h" from the test/ root. GTest::gtest for the
# AssertionResult it builds; calaman.iterative for the Info it reads.
calaman_add_interface_library(
  NAME
  calaman.test.shared.expect_converged
  LINK_PUBLIC
  calaman::iterative
  GTest::gtest
)
target_include_directories(
  calaman.test.shared.expect_converged
  INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/test>
)
