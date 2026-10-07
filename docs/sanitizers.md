# Sanitizer registry

Every sanitizer check this project turns off, narrows, or loosens, and every
check it never runs, in one place, so that nothing is off silently.

## The policy

- **A suppression is scoped to vendor code.** It names a vendor library, a
  vendor kernel, or a filter that selects only calaman's own code — never a
  pattern that could match a calaman or WarpWraps frame.
- **It is registered here**, with every field below filled, before it merges.
  A suppression, filter, or loosened threshold that is not in this file is a
  bug.
- **It is proven not to hide calaman code** — by a canary that must still fail
  with the suppression in place (see [Canaries](#canaries)). An entry whose
  *Proven by* reads "none yet" is unproven, and says so.

A **coverage gap** is a check that never runs at all. Gaps are listed too,
because an unchecked class of bug is as invisible in a green run as a
suppressed one.

## Fields

| Field | What it records |
|---|---|
| **What is off** | the exact setting, and where it lives |
| **Symptom** | what failed without it |
| **Observed on** | the versions it was seen on — the claim is only as fresh as these |
| **Why vendor-side** | why the finding is not ours (for a gap: why it is still open) |
| **Evidence** | the stack, kernel, or measurement that shows it |
| **Re-verify** | the change that should trigger checking it again |
| **Upstream** | the vendor report, if one was filed |
| **Proven by** | the canary that shows calaman code is still checked |

**Toolchain for every entry unless it says otherwise:** CUDA 13.0.88 (nvcc),
driver 580.126.20, compute-sanitizer 2025.3.1, NVIDIA RTX 3080 (`sm_86`); ROCm
7.2.4, AMD `gfx1200`; clang 20.1.8 with libc++. Dated 2026-10-06 (PR #228).

## Suppressions and narrowed scope

### S1. LSan: ROCm runtime leaks at exit

- **What is off:** `leak:libhsa-runtime64.so` and `leak:libamdhip64.so` in
  `test/shared/lsan.supp`, loaded by the `asan`, `hip-asan` and `ci-asan` test
  presets through `LSAN_OPTIONS` (`CMakePresets.json`).
- **Symptom:** `hip-asan` failed 94 suites on LeakSanitizer at process exit.
- **Observed on:** ROCm 7.2.4, gfx1200, clang 20.1.8.
- **Why vendor-side:** a fixed ~6KB per process that touches a device,
  independent of what the suite does; the HSA runtime frees its per-agent
  tables lazily and never at exit.
- **Evidence:** every frame of every reported leak stack is inside
  `libhsa-runtime64` or `libamdhip64`; no calaman or WarpWraps frame appears.
  The CUDA driver needs no entry.
- **Re-verify:** any ROCm bump — drop both lines and run `hip-asan`; delete
  them if it is green.
- **Upstream:** none yet.
- **Proven by:** `sanitizer_canary.lsan.leak` (`asan`, `hip-asan`): a leak
  whose stack is calaman's own is still reported with both lines loaded.
  Widening the file to `leak:*`, or `detect_leaks=0`, turns it red.

### S2. racecheck and synccheck scoped to calaman's kernels

- **What is off:** `--kernel-name kns=calaman` on the compute-sanitizer
  launcher when `CALAMAN_COMPUTE_SANITIZER_TOOL` is `racecheck` or `synccheck`
  (`CMakeLists.txt`, compute-sanitizer block). cuBLAS, cuSOLVER and every other
  vendor kernel go unchecked by those two tools. memcheck and initcheck stay
  unscoped.
- **Symptom:** racecheck unscoped spent >20 min in cuBLAS/cuSOLVER on a suite
  memcheck clears in 8s. synccheck unscoped reported a divergent barrier.
- **Observed on:** the default toolchain above (CUDA only).
- **Why vendor-side:** racecheck and synccheck judge a kernel's own shared
  memory and barriers, which in a vendor kernel we can neither read nor fix.
  The synccheck report is inside cuBLAS's own single-precision kernel; memcheck
  and initcheck are clean on the same call and the result matches the
  reference LAPACK.
- **Evidence:** synccheck: `trsm_ln_up_kernel` (float/float2 instantiations
  only), reached by the unit-triangular `wwr::trsm` at
  `src/lapack/sytrs2/sytrs2.cppm:290` and `src/lapack/hetrs2/hetrs2.cppm:285`.
  racecheck: wall time only, no finding.
- **Re-verify:** any CUDA toolkit or driver bump — run synccheck unscoped on
  the sytrs2/hetrs2 suites. The filter itself is only sound while every calaman
  `__global__` mangles `calaman`, which the build enforces: the
  `calaman_kernel_name_check` target (every CUDA build, `ALL`) runs
  `cuobjdump -symbols` over every `calaman_add_gpu_device_library` archive and
  fails on an entry that does not.
- **Upstream:** none yet.
- **Proven by:** `sanitizer_canary.compute_sanitizer.{racecheck_error,
  racecheck_warning,synccheck}`, kernels in `calaman::` that the filter must
  select. A filter that selects nothing of ours (`kns=nomatch`) turns them red,
  and a `__global__` outside `calaman::` fails the build.

### S3. `ASAN_OPTIONS=protect_shadow_gap=0`

- **What is off:** ASan's protection of the shadow gap, set in the `asan`,
  `hip-asan` and `ci-asan` test presets and the compose `asan` service
  (`docker/compose.yaml`).
- **Symptom:** under `asan`, every GPU suite died with
  `cudaErrorMemoryAllocation`.
- **Observed on:** the default toolchain above. Set on `hip-asan` too; whether
  ROCm needs it is not verified.
- **Why vendor-side:** the CUDA driver reserves virtual address ranges that
  fall in the gap ASan otherwise maps inaccessible. A wild access into the gap
  now lands in mapped memory instead of faulting.
- **Evidence:** the allocation failure disappears with this option alone, and
  `asan` then ran 208/208.
- **Re-verify:** any driver bump, and any clang bump that changes ASan's
  x86_64 memory layout. Also try `hip-asan` without it.
- **Upstream:** none — known ASan/CUDA interaction, not a bug.
- **Proven by:** `sanitizer_canary.asan.heap_buffer_overflow` (`asan`,
  `hip-asan`), which runs with this option set: instrumentation still fires.
  A build without `-fsanitize=address` turns it red.

### S4. `CALAMAN_TEST_TIMEOUT_MULTIPLIER` (loosened timing)

- **What is off:** no check — every test TIMEOUT is multiplied, 3 in `asan`,
  `hip-asan` and `ci-asan`, 10 in `compute-sanitizer` (`CMakePresets.json`;
  compose forwards `TIMEOUT_MULTIPLIER`). Listed because a hang hides longer
  behind it.
- **Symptom:** `SteqrOracleTests` overran its 120s TIMEOUT under `hip-asan` at
  1x.
- **Observed on:** ROCm 7.2.4, gfx1200 (the overrun); the factors were sized on
  both cards.
- **Why vendor-side:** not a vendor fault — instrumentation is slower by design.
  The multiplier is 1 in every non-sanitized preset.
- **Evidence:** the overrun above; PR #228's runs at these factors, 208/208.
- **Re-verify:** whenever a suite's TIMEOUT changes, or a sanitized run nears a
  multiplied limit.
- **Upstream:** n/a.
- **Proven by:** n/a — a timing allowance, not a filter.

### S5. The `no_sanitizer` ctest label

- **What is off:** any ctest entry labeled `no_sanitizer` is excluded by the
  `asan`, `hip-asan`, `ci-asan` and `compute-sanitizer` test presets and by
  both compose sanitizer services (`-LE no_sanitizer`); the pytest marker of
  the same name is deselected by the compose `compute-sanitizer` service.
- **Symptom:** none today — **nothing in calaman carries the label or the
  marker.** It is mechanism, kept so a case that instrumentation makes
  intractable is excluded by name rather than skipped.
- **Observed on:** n/a.
- **Why vendor-side:** n/a — adding the label is a suppression, and gets its own
  entry here with the reason.
- **Evidence:** `grep -rn no_sanitizer test src` is empty.
- **Re-verify:** any change that adds the label or the marker.
- **Upstream:** n/a.
- **Proven by:** n/a while unused.

## Coverage gaps

The fields read the same; for a gap, *Why vendor-side* says why it is open.

### G1. Closed: host code in `.cu` files is ASan-instrumented

- **What is off:** nothing. Kept under its number so citations still resolve.
  It was: under CUDA, `CALAMAN_ENABLE_ASAN` reached `CXX` units only, and every
  `.cu` (launch wrappers, functor setup) built without `-fsanitize`.
- **Symptom:** n/a — before #231, a host-side overflow in a `.cu` passed.
- **Observed on:** the default toolchain above (nvcc 13.0.88 with a clang
  20.1.8 host compiler); ROCm 7.2.4 for HIP.
- **Why vendor-side:** n/a. The fix: on CUDA, `CMAKE_CUDA_HOST_COMPILER`
  defaults to the CXX clang in every build, not only sanitized ones, so one
  ASan runtime serves both halves, and the ASan block passes
  `-Xcompiler=-fsanitize=address` / `-fno-omit-frame-pointer` to CUDA units
  (`CMakeLists.txt`). HIP never had the gap: a `.cu` there is a `CXX` unit
  (`-x hip`), so the CXX-gated flags already reached its host pass.
- **Evidence:** `build-asan/build.ninja` shows `-Xcompiler=-fsanitize=address`
  on every `.cu` compile, `build-hip-asan/build.ninja` `-fsanitize=address`;
  the canary below goes red (exit 0) when the CUDA flags are dropped.
- **Re-verify:** any CUDA bump — nvcc's supported clang range is per release —
  and any clang bump.
- **Upstream:** n/a.
- **Proven by:** `sanitizer_canary.asan.cu_heap_buffer_overflow`.

### G2. No device memory checking on AMD

- **What is off:** HIP device code runs unchecked: `hip-asan` instruments host
  code only, and `CALAMAN_COMPUTE_SANITIZER` refuses the HIP backend at
  configure.
- **Symptom:** none — an out-of-bounds device access on HIP is caught only if
  it corrupts a result the oracle compares.
- **Observed on:** ROCm 7.2.4, gfx1200.
- **Why vendor-side:** ROCm's device ASan needs XNACK, which gfx1200 does not
  offer (`rocminfo`: `XNACK enabled: NO`), and ROCm has no compute-sanitizer
  counterpart. Kernels shared by both backends are checked on CUDA.
- **Evidence:** `rocminfo`; the `FATAL_ERROR` in `CMakeLists.txt`.
- **Re-verify:** any ROCm bump or new AMD card.
- **Upstream:** n/a.
- **Proven by:** n/a.

### G3. Closed: memcheck runs with `--leak-check full`

- **What is off:** nothing. Kept under its number so citations still resolve.
  It was: memcheck ran without `--leak-check`, so a device allocation a suite
  never freed went unreported.
- **Symptom:** n/a.
- **Observed on:** the default toolchain above.
- **Why vendor-side:** n/a. The fix: the compute-sanitizer launcher appends
  `--leak-check full` when the tool is `memcheck` (`CMakeLists.txt`); a leak
  counts toward `--error-exitcode`, so it fails the suite.
- **Evidence:** with leak-check on, `compute-sanitizer` (memcheck) ran 242/242
  with no leak reported anywhere, so no calaman, test, or vendor allocation
  needed a fix or an entry here. The canary below exits 0 when the flag is
  dropped (2026-10-07, #232).
- **Re-verify:** any compute-sanitizer bump — the canary below does it.
- **Upstream:** n/a.
- **Proven by:** `sanitizer_canary.compute_sanitizer.memcheck_leak`.

### G4. No UBSan

- **What is off:** no `-fsanitize=undefined` build on either backend.
- **Symptom:** none — never run.
- **Observed on:** n/a.
- **Why vendor-side:** open, not vendor.
- **Evidence:** no UBSan option in `CMakeLists.txt` or `CMakePresets.json`.
- **Re-verify:** n/a.
- **Upstream:** n/a.
- **Proven by:** none yet. Tracked by #234.

### G5. No device sanitizer in CI (no GPU on hosted runners)

- **What is off:** CI checks host code only. Its `cpp (asan)` leg builds the
  `ci-asan` preset (`asan` with `ci-cuda`'s pinned `sm_86`) and runs
  `-LE gpu|no_sanitizer` under ASan+LSan, with the `asan` preset's
  `ASAN_OPTIONS`/`LSAN_OPTIONS` (S1, S3). Every `gpu`-labelled suite and the
  compute-sanitizer and `hip-asan` presets stay local-only, so device code,
  and host code reached only from a `gpu` suite, is never sanitized in CI.
- **Symptom:** a device memory error, or a host one on a GPU-only path, merges
  green unless someone ran a sanitized preset on a card.
- **Observed on:** GitHub-hosted `ubuntu-latest`, the `cuda-ci` image.
- **Why vendor-side:** hosted runners have no GPU, so no device code can run
  there at all; a HIP ASan leg would add a build for host code the CUDA leg
  already covers.
- **Evidence:** `ci.yml`'s `asan` matrix leg; it is part of `cpp`, so `ci-ok`
  requires it on every code change and accepts its skip only on a docs-only
  one.
- **Re-verify:** if a GPU runner is ever added — then run `compute-sanitizer`
  and `asan` with the `gpu` label included.
- **Upstream:** n/a.
- **Proven by:** `sanitizer_canary.asan.heap_buffer_overflow`,
  `sanitizer_canary.asan.cu_heap_buffer_overflow` and
  `sanitizer_canary.lsan.leak` run in that leg (none is `gpu`-labelled), so a
  CI build that loses `-fsanitize=address`, or an `LSAN_OPTIONS` that hides
  calaman's own leak, turns the leg red.

### G6. Closed: racecheck warning-severity hazards do fail a run

- **What is off:** nothing. Kept under its number so citations still resolve.
- **Symptom:** n/a.
- **Observed on:** compute-sanitizer 2025.3.1, RTX 3080.
- **Why vendor-side:** n/a — it was a question about the tool's contract.
- **Evidence:** a shared-memory race confined to one warp is reported as
  `Warning: Race reported` (`RACECHECK SUMMARY: ... 0 errors, 1 warning`) and
  the run exits 1 under `--error-exitcode 1`, the same as an ERROR-graded one.
- **Re-verify:** any compute-sanitizer bump — the canary below does it.
- **Upstream:** n/a.
- **Proven by:** `sanitizer_canary.compute_sanitizer.racecheck_warning`.

## Canaries

A canary is a deliberately buggy run that must fail under the check it guards
(`test/sanitizer_canaries/`, registered by `calaman_add_sanitizer_canary`). It
passes only when the run exits non-zero **and** prints the expected report, so
a crash for another reason, or a report the tool printed without failing the
run, is red. Each is configured only under its own preset and is absent from
every other; all carry the `sanitizer_canary` label.

| Canary | Preset | Defect | Expected report |
|---|---|---|---|
| `asan.heap_buffer_overflow` | `asan`, `hip-asan` | host write one past a `new[]` | `AddressSanitizer: heap-buffer-overflow` |
| `asan.cu_heap_buffer_overflow` | `asan`, `hip-asan` | host write one past a `new[]` in a `.cu` launch wrapper | `AddressSanitizer: heap-buffer-overflow` |
| `lsan.leak` | `asan`, `hip-asan` | host `new[]` dropped in `calaman::canary` | `LeakSanitizer: detected memory leaks` |
| `compute_sanitizer.memcheck` | `compute-sanitizer`, tool `memcheck` | global write one past the allocation | `Invalid __global__ write` |
| `compute_sanitizer.memcheck_leak` | tool `memcheck` | `wwrMalloc` never freed | `Leaked <n> bytes` |
| `compute_sanitizer.initcheck` | tool `initcheck` | read of never-written device memory | `Uninitialized __global__ memory read` |
| `compute_sanitizer.racecheck_error` | tool `racecheck` | cross-warp shared-memory race | `Error: Race reported` |
| `compute_sanitizer.racecheck_warning` | tool `racecheck` | intra-warp shared-memory race | `Warning: Race reported` |
| `compute_sanitizer.synccheck` | tool `synccheck` | `__syncthreads` reached by one warp of two | `Barrier error` |

Names are prefixed `sanitizer_canary.` in ctest. compute-sanitizer registers
only the selected tool's canaries, so covering all four is four reconfigures:

```bash
for t in memcheck initcheck racecheck synccheck; do
  cmake --preset compute-sanitizer -DCALAMAN_COMPUTE_SANITIZER_TOOL=$t
  cmake --build --preset compute-sanitizer
  ctest --preset compute-sanitizer -L sanitizer_canary
done
```
