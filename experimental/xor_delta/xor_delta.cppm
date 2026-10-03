/**
 * @file xor_delta.cppm
 * @brief The calaman.experimental.xor_delta module -- in-place XOR delta coding
 *        of a 64-bit integer stream.
 *
 * encode() replaces each element (past the first) with its XOR against the
 * previous ORIGINAL element, so a run of near-identical values collapses to
 * small deltas a downstream compressor packs well; decode() inverts it. The
 * first element is the baseline and is left untouched by both.
 *
 * The two loops run in OPPOSITE directions, and that is the whole correctness
 * argument: encode needs data[i-1] still at its pre-encoded value, so it walks
 * DOWN (i = count-1 .. 1) and touches data[i-1] only after data[i] is done;
 * decode has already-decoded data[i-1] to XOR against, so it walks UP.
 *
 * Pure host C++23: no BLAS, no device code, no WarpWraps -- it owns no .cu and
 * touches only `import std` (std::uint64_t, std::size_t). It lives under the
 * experimental/ tier, built only when CALAMAN_BUILD_EXPERIMENTAL is ON.
 *
 * Usage:
 *   import calaman.experimental.xor_delta;
 *   std::uint64_t v[]{10, 11, 11, 20};
 *   calaman::experimental::xor_delta_encode(v, 4);  // v now holds the deltas
 *   calaman::experimental::xor_delta_decode(v, 4);  // v back to {10,11,11,20}
 */

export module calaman.experimental.xor_delta;

import std; // std::uint64_t, std::size_t

export namespace calaman::experimental {

void xor_delta_encode(std::uint64_t* data, std::size_t count) {
  if (count <= 1) return;

  // Process backwards to modify the array in-place without overwriting
  // the previous un-differenced element needed for the next calculation.
  for (std::size_t i = count - 1; i > 0; --i) {
    data[i] = data[i] ^ data[i - 1];
  }
}

void xor_delta_decode(std::uint64_t* data, std::size_t count) {
  if (count <= 1) return;

  // Process forwards to restore values sequentially using the baseline.
  for (std::size_t i = 1; i < count; ++i) {
    data[i] = data[i] ^ data[i - 1];
  }
}

} // namespace calaman::experimental
