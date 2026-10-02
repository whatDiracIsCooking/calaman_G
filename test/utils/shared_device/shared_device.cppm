/**
 * @file shared_device.cppm
 * @brief One process-wide calaman::test::DeviceHandle, created on first use and
 *        torn down deterministically after the last test
 *
 * Every GPU suite needs a DeviceHandle to allocate a DeviceBufferWrapper
 * against, and constructing one queries the device and eagerly creates a stream
 * + pool (see calaman.test.shared.device_handle). Doing that per test case --
 * which most suites did, with a per-case std::make_shared<DeviceHandle>(0) --
 * repeats that setup for no benefit: one card, one process. shared_device()
 * hands every case in a process the SAME handle instead.
 *
 * Two properties matter, and neither is incidental:
 *
 *  - Lazy. The handle is created on the FIRST shared_device() call, not at
 *    process start. A binary that hosts a host-only suite alongside a GPU suite
 *    -- run GPU-less in CI via `ctest -LE gpu` -- must not touch a device when
 *    only the host suite runs, and a host suite never calls shared_device().
 *
 *  - Real teardown, not a leak. A plain function-local static would be destroyed
 *    at process exit (atexit), AFTER main returns and possibly after the GPU
 *    runtime has been torn down -- releasing a stream/pool against a dead
 *    runtime is the classic shutdown crash. The implementation parks the handle
 *    in a registered GoogleTest global environment whose TearDown() runs inside
 *    RUN_ALL_TESTS, before main returns, so the stream and pool are destroyed
 *    while the runtime is still alive.
 *
 * Usage:
 *   import calaman.test.utils.shared_device;
 *   auto handle = calaman::test::shared_device();   // same handle every call
 */

export module calaman.test.utils.shared_device;

// Re-exported: DeviceHandle is what the returned shared_ptr points at, so a
// caller that can name shared_device() can name and use what it returns (its
// stream(), pool(), props()) without a second import -- import is not
// transitive.
export import calaman.test.shared.device_handle;

import std;

export namespace calaman::test {

/// @brief The one DeviceHandle shared by every GPU case in this process
///
/// Creates it on the first call and returns that same handle thereafter; a
/// registered global test environment destroys it after the last test. Do not
/// call it from a host-only test -- doing so initialises the device.
std::shared_ptr<DeviceHandle> shared_device();

} // namespace calaman::test
