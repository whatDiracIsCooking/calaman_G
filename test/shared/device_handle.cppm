/**
 * @file device_handle.cppm
 * @brief Test-side device handle: one GPU's identity, properties, a default
 *        allocation stream and a default memory pool
 *
 * A DeviceBufferWrapper needs some concrete type satisfying WarpWraps's
 * device_handle ladder to back it, and the library ships none -- the handle is
 * the caller's to provide. This is the model the suites here allocate against:
 * the fullest tier (dev_idx() + stream() + pool()), so a buffer built on it
 * draws from that pool on that stream.
 *
 * Deliberately under test/, not src/. Nothing calaman SHIPS needs a handle:
 * calaman.lacpy takes a wwrStream_t and calaman.diff_norm a wwrblasHandle_t,
 * because neither allocates. Only the test tier allocates, so only the test tier
 * owns a handle -- and owning one here keeps the abort policy it hard-codes out
 * of calaman's public API. See test/shared/README.md.
 *
 * Move-only, because it owns a stream and a pool: two instances must never both
 * claim the same underlying stream or pool. An out-of-range index or a driver
 * failure aborts through AbortPolicy -- constructing one of these means you
 * intend to USE that device, which is not recoverable at this layer.
 *
 * Usage:
 *   import calaman.test.shared.device_handle;
 *   auto handle = std::make_shared<calaman::test::DeviceHandle>(0);
 */

export module calaman.test.shared.device_handle;

// Re-exported, because they are in this class's PUBLIC interface and an import
// is not transitive: props() returns wwr::wwrDeviceProp, and stream() / pool()
// return GpuStreamWrapper / GpuMemPoolWrapper. A consumer that can name
// DeviceHandle can therefore use everything it returns.
export import wwr.runtime_api;
export import wwr.extension.runtime;

// Plain imports: wwr.extension.handle backs only the static_assert below,
// wwr.extension.common only the private gpu_check call, and AbortPolicy is a
// private implementation detail of this fixture's stream and pool.
import wwr.extension.common;
import wwr.extension.handle;
import calaman.test.shared.abort_policy;
import std;

export namespace calaman::test {

/// @brief Identity, static properties, default stream and default pool of one GPU
///
/// Queries the device's properties once at construction and eagerly creates a
/// stream and a memory pool on it, so a caller can allocate device memory
/// without managing either. The full cudaDeviceProp / hipDeviceProp_t is held
/// directly and exposed through props(); individual fields are deliberately not
/// mirrored behind their own accessors.
class DeviceHandle {
  // stream_ and pool_ bind all three policy slots (create, destroy,
  // device-access) to abort-on-failure.
  using Abort = AbortPolicy<wwr::wwrError_t>;
  using Stream = wwr::extension::GpuStreamWrapper<Abort, Abort, Abort>;
  using Pool = wwr::extension::GpuMemPoolWrapper<Abort, Abort, Abort>;

public:
  explicit DeviceHandle(int index = 0,
                        std::source_location location = std::source_location::current())
      : index_(index), props_(query_props(index, location)),
        // The canonical DeviceBoundHandle ctor puts source_location after the
        // (defaulted) policy block, so forwarding `location` names the three
        // default policies first.
        stream_(index, {}, {}, {}, location), pool_(index, {}, {}, {}, location) {}

  int dev_idx() const noexcept { return index_; }

  /// @brief This device's static properties, queried once at construction
  const wwr::wwrDeviceProp &props() const noexcept { return props_; }

  /// @brief The default allocation stream, created on this device at construction
  Stream &stream() noexcept { return stream_; }
  /// @copydoc stream()
  const Stream &stream() const noexcept { return stream_; }

  /// @brief The default memory pool, created on this device at construction
  Pool &pool() noexcept { return pool_; }
  /// @copydoc pool()
  const Pool &pool() const noexcept { return pool_; }

private:
  /// @brief Query one device's properties, aborting on failure
  static wwr::wwrDeviceProp query_props(int index, std::source_location location) {
    wwr::wwrDeviceProp prop{};
    wwr::extension::gpu_check(wwr::wwrGetDeviceProperties(&prop, index), Abort{}, location);
    return prop;
  }

  int index_ = 0;
  wwr::wwrDeviceProp props_{};
  Stream stream_;
  Pool pool_;
};

/// The model is the fullest tier -- pinning it also pins the two tiers it
/// refines. Guards against an accessor later turning throwing or
/// non-const-callable, which would silently drop the buffer to a slower
/// allocation strategy instead of failing.
static_assert(wwr::extension::device_handle_pool<DeviceHandle>);

} // namespace calaman::test
