# test/utils/ -- reusable GPU test utilities, each built as a library the
# suites under test/ link against. Not test executables, and not shipped.
# Add a utility with add_subdirectory(<name>) when a suite needs one.
#
# shared_device: the one DeviceHandle per process (and the global environment
# that tears it down), shared across every GPU case so the device is queried and
# its stream/pool created once, not per test case.
add_subdirectory(shared_device)
