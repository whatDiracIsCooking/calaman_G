/**
 * @file workspace_builder.cppm
 * @brief The :workspace_builder partition of calaman.common -- size one device
 *        scratch buffer built from many aligned regions, and carve the pointers
 *
 * Pure host byte arithmetic: no device code, no allocation, no WarpWraps. A
 * routine that needs several device buffers wants ONE allocation carved into
 * aligned regions rather than many calls to the device allocator.
 *
 * Two kinds of region: FIXED regions are all live at the same time, so their
 * aligned sizes ACCUMULATE; SCRATCH regions are temporaries that alias, so only
 * the LARGEST is counted. total() is the fixed sum plus that largest scratch
 * region. Every region is rounded up to `alignment` (default 256, the base
 * alignment device allocators hand back), via the :align_up partition.
 *
 * Three names, outermost first -- prefer the outermost that fits:
 *   - carve_workspace + the SlicesFor concept: a routine's slices struct owns
 *     ONE carve() member describing its layout, and carve_workspace runs it
 *     over a null base to size (its *_bufferSize) or over the real base to
 *     carve, so the size query can never drift from the carving.
 *   - WorkspaceLayout: the primitive carve() members are written against.
 *   - WorkspaceBuilder: byte accounting only; the caller derives offsets itself.
 *
 * Deliberately not hardened against overflow: plain std::size_t arithmetic for
 * realistic workspace sizes, not adversarial inputs (see align_up.h).
 *
 * Usage:
 *   import calaman.common;   // the partition is reached through the whole module
 *
 *   struct Slices {          // one carve(), run twice -- see carve_workspace
 *     double *a = nullptr;
 *     int *piv = nullptr;
 *     double *work = nullptr;
 *     void carve(calaman::WorkspaceLayout &layout, const int n, const int lwork) {
 *       a = layout.fixed<double>(static_cast<std::size_t>(n) * n);
 *       piv = layout.fixed<int>(n);
 *       work = layout.scratch<double>(lwork); // scratch LAST, after every fixed
 *     }
 *   };
 *   const std::size_t bytes = calaman::carve_workspace<Slices>(nullptr, nullptr, n, lwork);
 *   Slices s;
 *   calaman::carve_workspace(d_work, &s, n, lwork); // same path, real pointers
 */

export module calaman.common:workspace_builder;

import std;

import :align_up; // calaman::align_up -- the per-region rounding

namespace calaman {

/// @brief Accumulate the byte size of one device workspace built from aligned
///        fixed and scratch regions
///
/// Add every buffer the workspace must hold, then read total() for the number of
/// bytes to allocate once. Fixed regions (add_fixed) coexist and sum; scratch
/// regions (add_scratch) alias, so only the largest is counted. See the file
/// header for the layout contract. All members are noexcept and do no allocation.
export class WorkspaceBuilder {
  std::size_t fixed_ = 0;   ///< running sum of the aligned fixed regions
  std::size_t scratch_ = 0; ///< largest aligned scratch region seen so far

public:
  /// @brief Reserve a fixed region that stays live -- its aligned size is added
  ///
  /// @tparam T Element type; each element is sizeof(T) bytes
  /// @param count Number of elements in one buffer
  /// @param copies Number of identical buffers to size (each aligned)
  /// @param alignment Byte boundary each buffer starts on (a power of two)
  template <typename T>
  void add_fixed(const std::size_t count, const std::size_t copies = 1,
                 const std::size_t alignment = 256) noexcept {
    fixed_ += align_up(count * sizeof(T), alignment) * copies;
  }

  /// @brief Reserve a scratch region that aliases others -- total tracks only the
  ///        largest such region, not their sum
  ///
  /// @tparam T Element type; each element is sizeof(T) bytes
  /// @param count Number of elements in one buffer
  /// @param copies Number of identical buffers to size (each aligned)
  /// @param alignment Byte boundary each buffer starts on (a power of two)
  template <typename T>
  void add_scratch(const std::size_t count, const std::size_t copies = 1,
                   const std::size_t alignment = 256) noexcept {
    scratch_ = std::max(scratch_, align_up(count * sizeof(T), alignment) * copies);
  }

  /// @brief Total bytes to allocate: every fixed region plus the largest scratch
  [[nodiscard]] std::size_t total() const noexcept { return fixed_ + scratch_; }
};

/// @brief Lay out ONE device workspace buffer into aligned regions, SIZING the
///        layout and handing back the typed pointer from the SAME calls
///
/// WorkspaceBuilder answers "how many bytes"; WorkspaceLayout also answers "where
/// is each region". Construct it over the real base to carve, or over nullptr to
/// size only -- every fixed()/scratch() returns nullptr in sizing mode but
/// advances the offsets identically, so a routine writes its layout once and runs
/// it twice: null base inside its *_bufferSize, the real base inside the routine.
/// The size query therefore cannot drift from the carving.
///
/// The two region kinds are WorkspaceBuilder's: FIXED regions coexist and
/// accumulate; SCRATCH regions alias, so they share one base and total() counts
/// only the largest. ADD EVERY FIXED REGION BEFORE ANY SCRATCH REGION -- the
/// scratch zone sits at the end of the fixed run, so a fixed() after a scratch()
/// would overlap it. Each region is rounded up to `alignment` (default 256, the
/// boundary a device allocator hands back, so a buffer from one starts aligned);
/// successive offsets stay aligned as long as every region shares that alignment.
/// Same overflow caveats as WorkspaceBuilder and align_up -- realistic workspace
/// sizes, not adversarial inputs.
export class WorkspaceLayout {
  std::byte *base_;         ///< buffer base, or nullptr to size without carving
  std::size_t fixed_ = 0;   ///< running end of the fixed run, in bytes
  std::size_t scratch_ = 0; ///< largest scratch region seen, in bytes

public:
  /// @brief Size only (@p base null) or carve from @p base.
  explicit WorkspaceLayout(void *const base) noexcept : base_(static_cast<std::byte *>(base)) {}

  /// @brief Reserve a fixed region of @p count elements of T; returns its aligned
  ///        base, or nullptr in sizing mode. The region stays live, so it
  ///        accumulates.
  template <typename T>
  [[nodiscard]] T *fixed(const std::size_t count, const std::size_t alignment = 256) noexcept {
    std::byte *const at = (base_ != nullptr) ? base_ + fixed_ : nullptr;
    fixed_ += align_up(count * sizeof(T), alignment);
    return reinterpret_cast<T *>(at);
  }

  /// @brief Reserve a scratch region of @p count elements of T; returns the base
  ///        of the shared scratch zone, or nullptr in sizing mode. Every scratch()
  ///        returns the SAME pointer -- the regions alias -- and total() reserves
  ///        only the largest. Call after every fixed(), never before.
  template <typename T>
  [[nodiscard]] T *scratch(const std::size_t count, const std::size_t alignment = 256) noexcept {
    scratch_ = std::max(scratch_, align_up(count * sizeof(T), alignment));
    return (base_ != nullptr) ? reinterpret_cast<T *>(base_ + fixed_) : nullptr;
  }

  /// @brief The address the next fixed() would return, WITHOUT reserving it -- the
  ///        base of a sub-layout a caller carves separately (e.g. expm's pade
  ///        region). nullptr in sizing mode.
  [[nodiscard]] void *cursor() const noexcept {
    return (base_ != nullptr) ? base_ + fixed_ : nullptr;
  }

  /// @brief Total bytes the layout spans: every fixed region plus the largest scratch.
  [[nodiscard]] std::size_t total() const noexcept { return fixed_ + scratch_; }
};

/// @brief A slices struct that lays itself out: default-constructible, with a
///        carve(WorkspaceLayout&, args...) member that is the ONLY description
///        of its layout
export template <typename S, typename... Args>
concept SlicesFor = std::default_initializable<S> &&
                    requires(S s, WorkspaceLayout &layout, const Args &...args) {
                      s.carve(layout, args...);
                    };

/// @brief Run S::carve over @p d_work -- a real base carves, a null base only
///        sizes -- and return the bytes the layout spans
///
/// The one code path behind a routine's *_bufferSize and its slices-carving
/// entry point, so the two cannot drift.
///
/// @param d_work Workspace base, or null to size without carving
/// @param out    Receives the carved slices; may be null (the sizing call)
export template <typename S, typename... Args>
  requires SlicesFor<S, Args...>
std::size_t carve_workspace(void *const d_work, S *const out, const Args &...args) {
  WorkspaceLayout layout(d_work);
  S s{};
  s.carve(layout, args...);
  if (out != nullptr) {
    *out = s;
  }
  return layout.total();
}

} // namespace calaman
