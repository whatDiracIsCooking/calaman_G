# test/shared/ -- test-support modules imported by more than one test target.
# Both are pure host module interfaces: no device code, no GoogleTest, so they
# carry neither a .cu nor an explicit-instantiation unit. Not shipped, and
# nothing under src/ may link them -- see this directory's README.md for why
# these two types live here rather than in src/.

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
  wwr.extension.common
  IMPORT_STD
)

# The concrete device_handle the suites allocate DeviceBufferWrapper against.
# wwr.extension.handle carries the device_handle ladder this satisfies,
# wwr.extension.runtime the GpuStreamWrapper / GpuMemPoolWrapper it owns, and
# wwr.runtime_api wwrDeviceProp / wwrGetDeviceProperties.
calaman_add_cxx_module_library(
  NAME
  calaman.test.shared.device_handle
  PRIMARY_INTERFACE
  device_handle.cppm
  LINK_PUBLIC
  wwr.runtime_api
  wwr.extension.common
  wwr.extension.handle
  wwr.extension.runtime
  calaman.test.shared.abort_policy
  IMPORT_STD
)
