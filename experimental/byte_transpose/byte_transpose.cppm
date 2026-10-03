/**
 * @file byte_transpose.cppm
 * @brief The calaman.experimental.byte_transpose module -- regroup an array of
 *        64-bit values by byte significance, and the inverse that restores it.
 *
 * byte_transpose() takes `n` contiguous std::uint64_t at `inp` and writes, into
 * the `n`-value (8*n byte) buffer `out`, the same bytes reordered into eight
 * equal blocks by significance: block 0 is the most-significant byte of every
 * value, block 7 the least. A "structure of byte planes" layout -- the shape a
 * byte-oriented compressor or SIMD transform wants, since like-significance
 * bytes (which vary slowly across a sorted or near-constant array) end up
 * adjacent. byte_untranspose() is its exact inverse.
 *
 * Endianness-independent by construction: bytes are extracted from `inp` by
 * value (shift + mask), never by reinterpreting its storage, and reassembled
 * the same way, so the block layout and the roundtrip are identical on any
 * host. `out` is addressed as a byte buffer (std::byte aliases any object), so
 * `inp` and `out` must not overlap.
 *
 * Pure host C++23: no BLAS, no device code, no WarpWraps -- it owns no .cu and
 * touches only `import std` (std::uint64_t, std::size_t, std::byte). It lives
 * under the experimental/ tier, built only when CALAMAN_BUILD_EXPERIMENTAL is
 * ON.
 *
 * Usage:
 *   import calaman.experimental.byte_transpose;
 *   std::uint64_t v[]{...};                           // n values
 *   std::uint64_t planes[std::size(v)];               // 8*n bytes of planes
 *   calaman::experimental::byte_transpose(v, planes, std::size(v));
 *   std::uint64_t back[std::size(v)];
 *   calaman::experimental::byte_untranspose(planes, back, std::size(v));
 */

export module calaman.experimental.byte_transpose;

import std; // std::uint64_t, std::size_t, std::byte

export namespace calaman::experimental {

// Scatter the eight bytes of each value into significance-grouped planes:
// out byte [block*n + i] holds byte (7 - block) of value i, so block 0 is every
// value's MSB and block 7 every value's LSB. `out` must hold n values (8*n
// bytes) and must not overlap `inp`.
void byte_transpose(const std::uint64_t* inp, std::uint64_t* out,
                    std::size_t n) {
  auto* planes = reinterpret_cast<std::byte*>(out);
  for (std::size_t block = 0; block < 8; ++block) {
    const unsigned shift = 8u * (7u - static_cast<unsigned>(block));
    std::byte* plane = planes + block * n;
    for (std::size_t i = 0; i < n; ++i)
      plane[i] = static_cast<std::byte>((inp[i] >> shift) & 0xFFu);
  }
}

// Inverse of byte_transpose: gather the significance-grouped byte planes at
// `inp` (8*n bytes) back into n values at `out`. `out` must not overlap `inp`.
void byte_untranspose(const std::uint64_t* inp, std::uint64_t* out,
                      std::size_t n) {
  const auto* planes = reinterpret_cast<const std::byte*>(inp);
  for (std::size_t i = 0; i < n; ++i)
    out[i] = 0;
  for (std::size_t block = 0; block < 8; ++block) {
    const unsigned shift = 8u * (7u - static_cast<unsigned>(block));
    const std::byte* plane = planes + block * n;
    for (std::size_t i = 0; i < n; ++i)
      out[i] |= static_cast<std::uint64_t>(
                    std::to_integer<std::uint8_t>(plane[i]))
                << shift;
  }
}

} // namespace calaman::experimental
