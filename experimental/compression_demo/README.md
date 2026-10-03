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

For a run on a **real, production-shaped RI-DF matrix** (a multi-GB HDF5 CDERI
tensor), see "Real RI-fit CDERI data" below — it is a separate driver
(`cderi_demo.py`) because the dataset is far too large to live in `datasets()`
alongside the 512 KiB synthetic arrays.

## Where the data comes from

- **Synthetic patterns** — near-constant, monotonic/sorted, linspace, noisy,
  small-ints — isolate which data *shapes* the pipeline helps.
- **`geqp3: *`** — the **real** matrices the geqp3 test suite feeds the device.
  `geqp3_dump.cpp` reuses that suite's exact generator (`std::mt19937(seed)` +
  `uniform_real_distribution(-3,3)`, column-major, + `make_rank_deficient`),
  compiled with the project's clang-20 + libc++, so the streams are bit-identical
  to the tests' — only scaled to 256–512 sizes, since the suite's own shapes
  (≤16) are too small for a meaningful ratio.
- **RI-fit CDERI** — a real metric-contracted Cholesky 3-center integral tensor
  from PySCF (see below). `.h5` inputs are git-ignored (multi-GB, regenerable
  from PySCF).

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

## Bit-plane transpose (`bit_transpose_demo.py`)

`byte_transpose` groups by *byte* significance (8 planes); `bit_transpose` goes one
level finer — 64 planes, plane `b` holding bit `b` of every value, bit-packed (same
total size when `n % 8 == 0`). The idea: an individual low-entropy bit (a constant
sign bit, or a high exponent bit sharing a byte with varying neighbours) becomes a
contiguous run of identical bits the codec collapses, which byte-planes can't isolate.

```bash
uv run --with numpy --with cramjam \
    experimental/compression_demo/bit_transpose_demo.py
```

It is a **sharper tool that cuts both ways** (zstd):

| dataset | byte T | bit T | winner |
|---|---|---|---|
| linspace | 37.4 | **142.7** | bit (≈4×) |
| near-constant | 2.12 | **2.42** | bit |
| geqp3 dense | **1.15** | 1.10 | byte |
| geqp3 rankdef_r64 | **9.01** | 8.79 | byte |

- **bit-transpose wins on smooth / structured data** (linspace, near-constant, sorted
  — where bit positions have exploitable structure).
- **it loses or ties on the real geqp3 (high-entropy) matrices**, and under **LZ4** it
  actively *hurts* geqp3 (1.06 → 1.00): bit-packing destroys the byte-aligned runs
  LZ4's match-finder needs. It's far safer under **zstd** than LZ4/Snappy.

So for this project's workload `byte_transpose` stays the default (as good or better,
at 1/8 the planes and no bit-packing); `bit_transpose` is a lever for smooth fp64
(time series, parameter sweeps) under a strong codec — it reinforces rather than
displaces the Phase-2 recommendation.

### XOR-delta commutes with transpose — exactly

Order is irrelevant for the XOR-delta + transpose pair: `transpose(xor_delta(x))` and
`xor_delta_within_planes(transpose(x))` produce **bit-for-bit identical** output (hence
identical ratio), verified across all datasets for both byte and bit transpose. The
reason is algebraic — they act on orthogonal axes:

- transpose is a *value-independent bit permutation* π (bit `j` of element `i` moves to
  a slot fixed by `j`, keeping element `i`'s identity);
- xor_delta is a *per-element bitwise XOR*, and XOR is carry-free, so
  `bitⱼ(xᵢ ⊕ xᵢ₋₁) = bitⱼ(xᵢ) ⊕ bitⱼ(xᵢ₋₁)` — thus `π(xᵢ ⊕ xᵢ₋₁) = π(xᵢ) ⊕ π(xᵢ₋₁)`.

**Caveat:** this is special to XOR. An *arithmetic* (subtraction) delta would **not**
commute — its carries cross byte/bit boundaries. Practical upshot: a Phase-2
implementation may order xor_delta and transpose purely for performance; the
compressed result is guaranteed the same.

## Real RI-fit CDERI data (`cderi_demo.py`)

The synthetic + geqp3 arrays above are small and either random or structured by
construction. `cderi_demo.py` runs the *same* transforms + codecs over a real
density-fitting tensor: the metric-contracted Cholesky 3-center integrals PySCF
writes to disk (`L^P_ij`, shape `(naux, npair)`, fp64), the exact object a
RI/DF-based solver consumes. The reference run used `gly10` (glycine
decapeptide, cc-pVDZ / cc-pVDZ-rifit): `(2744, 269745)` = **5.92 GB**.

```bash
uv run --with numpy --with cramjam --with h5py \
    experimental/compression_demo/cderi_demo.py
```

It reuses the demo's faithful (C++-pinned) transforms, but swaps in a
memory-safe plane-wise `byte_transpose` — the demo's vectorised version builds an
`(8, n)` uint64 intermediate, which is ~47 GB at this `n` — and asserts it is
bit-identical to the tested one on a slice before using it.

### What the CDERI numbers say — the transforms barely matter; sparsity does

The matrix is **57% exact zeros** (`nnz-frac 0.43`), values in `[-0.71, 1.93]`
(RI-V metric contraction drives most AO-pair blocks to zero). That one fact sets
the whole result: it is ~2.2–2.3× compressible *by the codec alone*, and no
byte-level transform meaningfully improves on that.

| codec | raw | xor | transpose | xor+transpose |
|---|---|---|---|---|
| LZ4 | **2.23** | 2.13 | 2.15 | 2.09 |
| Snappy | **2.10** | 2.02 | 1.97 | 1.93 |
| zstd | 2.28 | 2.25 | **2.29** | 2.20 |

- **`raw` wins under LZ4 and Snappy**; `byte_transpose` edges it only under zstd,
  and only by +0.4% (2.29 vs 2.28). Unlike the *dense* geqp3 matrices — where
  transpose was the single robust winner — here the compressible structure is
  the zero-runs, which a byte transform *scatters* across planes rather than
  concentrates. So the transform's one home turf (dense high-entropy fp64) is
  exactly what this data is not.
- **`xor_delta` hurts everywhere**, as on the geqp3 data — its sequential-delta
  assumption fights the sparsity, and the combination is worst of all.

### Scaling and the frexp split don't help here either (zstd)

|  base T | pow2+raw | pow2+xor | pow2+T | pow2+xorT | maxabs+T | mant/exp split |
|---|---|---|---|---|---|---|
| 2.29 | 2.28 | 2.25 | 2.29 | 2.20 | 2.29 | 2.19 |

Lossless `/2^k` is confirmed exact (`pow2 exact? = yes`) and a complete wash
(2.29 → 2.29), matching the synthetic finding — a constant exponent offset is
free but buys nothing. The lossy `maxabs` is also 2.29 (no help, and it costs
~1 ULP). The per-element frexp mant/exp split is lossless but a **net loss**
(2.29 → 2.19, mantissa = 89% of the bytes): as on every other dataset, it merely
relocates exponent bits the codec already handles and pays a second-stream tax.

**Takeaway, reinforcing Phase 2:** this real RI-DF tensor confirms the thesis on
production-scale data — the compression comes from *data structure* (here
sparsity, as rank-deficiency drove the geqp3 win), which the codec captures with
no transform; `byte_transpose` is at best a zstd-only tie-breaker and `xor_delta`
is a liability. For sparse DF integrals specifically, a sparsity-aware path (drop
the structural zeros before the codec) is the obvious next lever — measured
directly below. zstd again leads LZ4/Snappy (~2.28 vs 2.23).

### Stripping the exact zeros first (`cderi_nonzero_demo.py`)

The in-place result hands the 57% zeros to the codec inline. The complementary
move is to **strip them first** and compress only the dense nonzero stream — but
losslessly, which means also storing a *presence bitmask* (1 bit/element) to put
the zeros back. `cderi_nonzero_demo.py` measures both halves:

```bash
uv run --with numpy --with cramjam --with h5py \
    experimental/compression_demo/cderi_nonzero_demo.py [path/to/cderi.h5]
```

**The nonzero values are essentially incompressible** — the zeros were the only
structure. Intrinsic ratio of the 2.56 GB nonzero stream (vs its own size):

| codec | raw | xor | transpose | xor+transpose |
|---|---|---|---|---|
| LZ4 | 1.00 | 1.00 | 1.07 | **1.08** |
| Snappy | 1.00 | 1.00 | 1.06 | **1.06** |
| zstd | 1.00 | 1.02 | **1.11** | 1.10 |

**But splitting the zeros out still wins end-to-end**, because the *bitmask*
compresses spectacularly. Lossless ratio vs the original 5.92 GB,
`orig / [compress(values) + compress(mask)]`:

| codec | values | mask | total | end-to-end | in-place (above) |
|---|---|---|---|---|---|
| LZ4 | 2.38 GB | 11.3 MB | 2.39 GB | **2.48×** | 2.23× |
| Snappy | 2.40 GB | 15.1 MB | 2.42 GB | **2.45×** | 2.10× |
| zstd | 2.31 GB | **0.1 MB** | 2.31 GB | **2.56×** | 2.29× |

- **Sparse-split beats in-place on every codec** (+11% LZ4, +17% Snappy, +12%
  zstd). The headline: the 92.5 MB presence mask crushes to **0.1 MB under zstd
  (~925×)** — the zero pattern is highly structured (the packed lower-tri AO-pair
  sparsity repeats across the 2744 aux vectors), so isolating it lets the entropy
  coder exploit it fully. LZ4/Snappy manage only 11–15 MB on the same mask, which
  is why zstd's margin widens.
- **The win is bounded, and modestly so** — the earlier prediction that a
  sparsity path would "dwarf" byte transforms was too strong. You still must
  store ~2.56 GB of high-entropy nonzero doubles that *no* lossless transform
  shrinks (~1.0–1.1×), so the lossless ceiling sits near the zero fraction
  (~2.3–2.6×). Beating it needs a *lossy* attack on the values: fp32 downcast of
  the nonzero stream (per `lossy_downcast.py`, ~2× at 6e-8 error) stacks on top
  for ~5× total.

### fp32 downcast confirms ~5×; pre-sorting the values does *not* help (`cderi_fp32_sort_demo.py`)

Two follow-ons to the lossy lever above, both end-to-end vs the original 5.92 GB
(lossy only in the fp32 cast — rms rel err **6.5e-6**, 0 underflow; a tail of
near-denormal values ~1e-40 casts to fp32 subnormals with max rel err ~7e-2, but
their absolute error is ~1e-41, negligible):

```bash
uv run --with numpy --with cramjam --with h5py \
    experimental/compression_demo/cderi_fp32_sort_demo.py [path/to/cderi.h5]
```

**fp32 downcast alone lands the predicted ~5×** — strip zeros, cast the nonzero
stream to fp32, compress, keep the bitmask: **5.12× zstd** (4.69× LZ4, 4.58×
Snappy), a clean 2× over the #123 lossless 2.56×.

The tempting next step is to **sort** the nonzero values first, so neighbouring
magnitudes cluster and `xor+transpose` gets a smooth, near-monotonic sequence to
delta-code. It works *spectacularly* on the values — but loses the gain to the
side-channel it forces, because restoring the original order needs the full sort
permutation, which is **incompressible**:

| codec | fp32 values (sorted, intrinsic) | sort permutation (uint32, 1.28 GB) | **A: fp32 in-place** | **B: fp32 sorted** |
|---|---|---|---|---|
| LZ4 | 4.01× | 1.00× | **4.69×** | 3.68× |
| Snappy | 3.38× | 1.00× | **4.58×** | 3.54× |
| zstd | **5.50×** | 1.04× | **5.12×** | 4.04× |

- **Sorting makes the fp32 value stream 3.4–5.5× compressible** (vs ~1.0–1.1×
  unsorted) — the monotonic-sequence case where `xor_delta` finally earns its
  keep, exactly as on synthetic `linspace`.
- **But an arbitrary order over 320M elements costs ~n·log₂n bits** (~1.23 GB
  even after zstd), and that *exceeds* the compression it buys. Under zstd,
  sorting shrinks the values by 0.93 GB (1.16→0.23 GB) but the permutation adds
  1.23 GB — a net **+0.30 GB**, dropping 5.12× → **4.04×**.

**Takeaway:** for a general lossless-position CDERI store, **don't pre-sort —
just fp32-downcast.** Sorting only pays when the permutation is cheap (data
already near-sorted, or a use that doesn't need the original order restored),
which density-fitting integrals are not. The honest ranking on this tensor:
fp32-in-place (5.12×) > fp32-sorted (4.04×) > fp64 lossless zero-strip (2.56×).

## Files

| File | What |
|---|---|
| `compression_demo.py` | the lossless spike: numpy transforms + `_selfcheck` + ratio/scaling/split tables |
| `cderi_demo.py` | the same transforms + codecs over a real multi-GB PySCF RI-fit CDERI `.h5` |
| `cderi_nonzero_demo.py` | strip exact zeros, compress the dense stream + a presence bitmask; sparse-split vs in-place |
| `cderi_fp32_sort_demo.py` | fp32 downcast of the nonzero stream, and whether pre-sorting pays once the permutation is stored (it doesn't) |
| `lossy_downcast.py` | lossy fp32/fp16 downcast, global vs per-element (block-FP) scaling |
| `bit_transpose_demo.py` | bit-plane vs byte-plane transpose + xor/transpose commutativity |
| `geqp3_dump.cpp` | standalone dumper of the geqp3 suite's matrices to raw f64 |
| `data/*.f64` | generated column-major fp64 dumps (git-ignored) |
| `*.h5` | real RI-fit CDERI input tensors read by `cderi_demo.py` (git-ignored, multi-GB) |
