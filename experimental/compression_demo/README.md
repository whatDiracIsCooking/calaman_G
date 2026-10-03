# compression_demo — Phase-1 ratio spike

Does `xor_delta_encode → byte_transpose → {LZ4, Snappy}` improve the compression
ratio of fp64 data? This is a **numpy mirror** of
`experimental/{xor_delta,byte_transpose}` (it validates the *idea*, not the
shipped C++), pinned faithful to the C++ layout contract by `_selfcheck()`.

```bash
# 1. dump the real geqp3-path matrices (clang-20 + libc++, same as the project)
clang++-20 -std=c++23 -stdlib=libc++ -O2 \
    experimental/compression_demo/geqp3_dump.cpp -o /tmp/geqp3_dump
/tmp/geqp3_dump experimental/compression_demo/data

# 2. run the spike (no uv.lock changes -- deps are ephemeral via --with)
uv run --with numpy --with cramjam \
    experimental/compression_demo/compression_demo.py
```

The `.f64` dumps are git-ignored (regenerable, ~9 MB); step 1 recreates them. If
`data/` is empty the demo still runs on the synthetic datasets and says so.

## Where the data comes from

- **Synthetic patterns** — near-constant, monotonic/sorted, linspace, noisy,
  small-ints — isolate which data *shapes* the pipeline helps.
- **`geqp3: *`** — the **real** matrices the geqp3 test suite feeds the device.
  `geqp3_dump.cpp` reuses that suite's exact generator (`std::mt19937(seed)` +
  `uniform_real_distribution(-3,3)`, column-major, + `make_rank_deficient`),
  compiled with the project's clang-20 + libc++, so the streams are bit-identical
  to the tests' — only scaled to 256–512 sizes, since the suite's own shapes
  (≤16) are too small for a meaningful ratio.

## What the numbers say (LZ4, Snappy, and zstd as a strong-codec upper bound)

- **`byte_transpose`-alone is the single robust winner.** It is best or tied on
  nearly every dataset across all three codecs — grouping like-significance bytes
  is what the entropy coder exploits. On geqp3 dense fp64 it is the *only* thing
  that beats the noise floor (zstd 1.15× vs 1.04× raw).
- **`xor_delta` doesn't just fail to help — it actively hurts the stronger the
  codec gets.** Under zstd, `xor+transpose` is consistently *worse* than
  transpose-alone (geqp3 dense 1.12× vs 1.15×, `rankdef_r64` 8.76× vs 9.01×,
  small-ints 6.21× vs 6.82×, even `linspace` 81× vs **176×** for xor-alone). A
  strong entropy coder would rather see the transposed planes than XOR-scrambled
  deltas. On real geqp3 data `xor_delta` ≈ raw regardless of codec — its
  sequential-delta assumption doesn't match column-major LA data.
- **Dense fp64 is essentially incompressible.** geqp3 dense tops out at ~1.15×
  (zstd + transpose); LZ4/Snappy sit at the ~1.08× Gaussian-noise floor. The real
  compressibility in LA data comes from **rank deficiency** (geqp3 `rankdef_r64`
  7.55× raw on LZ4, 8.19× on zstd), which the codec already captures *without* any
  transform; transpose nudges it to 8.0–9.0×.
- **Both transforms can lose to raw.** Snappy on `rankdef_wide` (raw 3.94× →
  transpose 2.76×) is the clearest case.

## Pre-scaling into [-1, 1] — lossless, but not a compression lever

Scaling every element by `1/2^k` (`k` = smallest power of two ≥ max|arr|) lands all
values in (-1, 1] and is **bitwise lossless** — division by a power of two only
decrements the exponent field, leaving the sign and all 52 mantissa bits intact.
The demo confirms this: `pow2 exact? = yes` on every dataset. (Dividing by a
*general* `|M|` instead is a real multiply — ≈½ ULP error, `maxabs relerr ~1e-16`.)

Crossing lossless `/2^k` with *every* stage (not just transpose) shows it **does
not meaningfully help any of them** (zstd; `base T` = transpose, no scale):

| dataset | base T | pow2+raw | pow2+xor | pow2+T | pow2+xorT | maxabs+T (lossy) |
|---|---|---|---|---|---|---|
| geqp3 dense | 1.15 | 1.05 | 1.05 | **1.17** | 1.14 | 1.15 |
| geqp3 rankdef_r64 | 9.01 | 8.28 | 8.26 | **9.17** | 8.98 | 9.04 |
| small ints | 6.82 | 4.46 | 4.01 | **6.87** | 6.25 | 3.49 |
| linspace | 37.4 | 1.36 | 175.1 | 33.6 | 86.2 | 96.6 |

Reading the row: `pow2+T` ≈ `base T` everywhere (the marginal +1–2% lossless win);
`pow2+xor` ≈ `pow2+raw` on real geqp3 data (scaling does **not** rescue
`xor_delta`); and `pow2+xorT` stays *below* `pow2+T` (adding `xor_delta` to a
scaled array still drags the combined pipeline down). The one exception is
`linspace`, where `xor_delta` dominates with or without scaling — the same smooth-
sequence special case as before.

Why `/2^k` is a wash: it applies the *same* exponent offset to every element, so it
preserves dynamic range exactly, leaves all mantissa byte-planes bitwise-identical,
and only shifts the exponent plane by a constant — which an entropy coder packs the
same. The ±1–2% wiggle is a second-order effect (the constant shift re-aligns the
11-bit exponent across the top-byte/second-byte boundary), and it goes *negative*
on `linspace`. The lossy `/|M|` scrambles the mantissas, so it swings wildly —
sometimes much better (linspace, near-constant), sometimes much worse (small ints,
which it turns from clean integers into rounded junk) — and costs ~1 ULP. Neither
is a reliable win for a single array, with any stage.

### Per-element scaling into [0.5, 1) — the frexp split (also a no-op here)

A *global* `/2^k` only pins each array's **max** to [0.5, 1); smaller elements fall
below 0.5, so it can't homogenize the exponent field. The only way to put **every**
element in [0.5, 1) is per-element: `frexp` each double into a mantissa ∈ [0.5, 1)
and an integer exponent, then compress the two streams separately (mantissa via
`byte_transpose`, exponents as int16). That *does* make the mantissa stream's
exponent field constant — but it's **lossless yet a slight net loss** vs plain
transpose (zstd):

| dataset | base T | mant/exp split |
|---|---|---|
| geqp3 dense | 1.15 | 1.14 |
| geqp3 rankdef_r64 | 9.01 | 8.93 |
| small ints | 6.82 | **5.93** |
| near-constant | 2.12 | 1.98 |

It loses because `byte_transpose` *already* isolates the exponent bits into the top
plane and compresses them well — the split merely *relocates* the same information
(it's entropy-conserving), adds a second-stream overhead, and on `small ints`
scrambles the clean integer bit-patterns zstd was exploiting.

### Where scaling *would* pay (and why not here)

The one regime scaling helps is a **corpus of narrow-dynamic-range arrays**:
normalize each so its whole value band lands in the same place, and the top
exponent byte becomes identical *across* arrays, so a compressor spanning the
concatenation collapses that plane (lossless via `/2^k`). The catch — **our geqp3
matrices have wide dynamic range** (uniform(-3,3), rank-deficient combos span many
exponents), so aligning the max leaves the exponents spread and nothing collapses.
The trick is real for checkpoints / batched tiles of tightly-ranged data; it is
*not* a lever for the data this project produces.

**Takeaway for Phase 2:** if we promote this to a C++ `experimental/compression`
tier, ship **`byte_transpose` as the workhorse** and treat the full
`xor+transpose` pipeline with suspicion — `xor_delta` earns its keep *only* on
smooth sequences with a weak codec, and even there a strong codec prefers
transpose-alone or xor-alone, never the combination. So: make `xor_delta`
optional/off-by-default, keep the pipeline a *choice* (a "raw beats both" case is
real), and prefer **nvCOMP** (already in the CUDA image, GPU, on-identity) over
adding CPU LZ4/Snappy as a new host dependency. zstd's ~15–20% edge over LZ4 is
worth noting if nvCOMP's zstd path is available on both backends.

## Lossy downcasting (`lossy_downcast.py`)

A *lossy* extension: scale, then cast `fp64 -> {fp32, fp16}` before the usual
transform + codec, round-trip back to fp64, and measure both ratio and error. The
scale/unscale is exact (power-of-two exponent shift), so the **only** lossy step is
the cast — error floors are fp32 `2^-24` (5.96e-8) and fp16 `2^-11` (4.88e-4).
Ratio is vs the **original fp64** byte count, so the width reduction (2× / 4×) is
part of the win.

```bash
uv run --with numpy --with cramjam \
    experimental/compression_demo/lossy_downcast.py
```

### fp32 is a clean ~2× over the best lossless pipeline

| dataset (zstd, transpose) | f64 lossless | fp32 | fp16 |
|---|---|---|---|
| geqp3 dense | 1.15 | 2.35 | 4.92 |
| geqp3 rankdef_r64 | 9.01 | 18.63 | 36.83 |
| noisy gaussian | 1.13 | 2.32 | 4.35 |

Each precision halving roughly doubles the ratio; the stage ranking is unchanged
(`transpose` wins, `xor_delta` helps only on smooth data). fp32 round-trips at a
uniform **6e-8 with zero underflow** — its 8 exponent bits have ample headroom
below 1.

### fp16 is dangerous under *global* scaling — and per-element fixes it

Global `/2^k` scaling is **wrong for fp16**: its 5-bit exponent bottoms out at
~6.1e-5, so pinning the max below 1 drags small elements into subnormals/underflow.
On the wide-dynamic-range geqp3 matrices the max relative error blows up to **1.0**
(e.g. `rankdef_r256`: max 3262, min 3.2e-9, span ~10¹² — 4 elements underflow to 0).

**Per-element (block-floating-point) scaling** — `frexp` each value into a mantissa
∈ [0.5,1) plus its own integer exponent, downcast the mantissa, store exponents as a
second int16 stream — removes the failure mode entirely: the mantissa is always a
*normal* fpN, so there is no underflow and the error is a uniform ~roundoff:

| fp16, zstd, transpose | global maxerr / uflow | per-element maxerr / uflow |
|---|---|---|
| geqp3 rankdef_r256 | **1.00** / 4 | **4.9e-4** / 0 |
| geqp3 rankdef_r64 | 0.31 / 0 | 4.8e-4 / 0 |
| geqp3 dense | 1.35e-2 / 0 | 4.9e-4 / 0 |

The cost is ~10% ratio (the exponent side-stream, which compresses well). Its one
downside: data sitting on a power-of-two boundary fragments into two exponent
classes (`near-constant` 20165× → 17.6×). fp32 needs *neither* mode — global
already gives uniform 6e-8.

**Takeaway:** fp32 downcast is the biggest single lever found (~2× over lossless,
bounded 6e-8 error). fp16 doubles that again but only via per-element
block-floating-point, and only where a ~5e-4 relative error is acceptable; raw
global-scaled fp16 is unsafe on general LA output.

## Files

| File | What |
|---|---|
| `compression_demo.py` | the lossless spike: numpy transforms + `_selfcheck` + ratio/scaling/split tables |
| `lossy_downcast.py` | lossy fp32/fp16 downcast, global vs per-element (block-FP) scaling |
| `geqp3_dump.cpp` | standalone dumper of the geqp3 suite's matrices to raw f64 |
| `data/*.f64` | generated column-major fp64 dumps (git-ignored) |
