// Implementation of calaman.test.utils.shared_device: the lazily-created handle
// and the GoogleTest global environment that tears it down before the GPU
// runtime shuts down. gtest is a textual header (not a module), so it is pulled
// in through the global module fragment -- the same `#include <gtest/gtest.h>`
// plus `import std;` mix every suite TU already uses.

module;

#include <gtest/gtest.h>

module calaman.test.utils.shared_device;

import std;

namespace calaman::test {
namespace {

// The shared handle lives in a function-local static -- not a namespace global
// -- so it is created on first use, destroyed only through the environment
// below, and never reported as mutable global state. Null until the first
// shared_device() call.
std::shared_ptr<DeviceHandle> &device_slot() {
  static std::shared_ptr<DeviceHandle> handle;
  return handle;
}

// TearDown() -- not a destructor at process exit -- is what makes the teardown
// "real": GoogleTest runs it inside RUN_ALL_TESTS, before main returns, so the
// stream and pool DeviceHandle owns are released while the runtime is still
// alive. SetUp() is intentionally the base no-op: the handle is created lazily
// by shared_device(), so a host-only suite sharing this binary never constructs
// one and `ctest -LE gpu` on a card-less runner stays device-free.
class SharedDeviceEnvironment : public ::testing::Environment {
public:
  void TearDown() override { device_slot().reset(); }
};

// Registered during static initialisation, before main and thus before
// RUN_ALL_TESTS. This unit is linked into a binary iff that binary references
// shared_device() below (every GPU suite does); a host-only-only binary neither
// links this unit nor registers the environment. GoogleTest takes ownership of
// the heap Environment -- its API has no non-owning overload, hence the raw new.
[[maybe_unused]] const ::testing::Environment *const kEnvironmentRegistered =
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    ::testing::AddGlobalTestEnvironment(new SharedDeviceEnvironment);

} // namespace

std::shared_ptr<DeviceHandle> shared_device() {
  auto &handle = device_slot();
  if (!handle) {
    handle = std::make_shared<DeviceHandle>(0);
  }
  return handle;
}

} // namespace calaman::test
