# Architecture notes

Decisions this project made, and facts about LAPACK, the vendor libraries or C++
that it has to live with — the things a reader will otherwise rediscover the hard
way.

Headers under `src/` state rules; this file says why, and on what it was
measured. That separation exists because a header claims to describe code as it
is *now* — a sentence like "verified on ROCm 7.2.4" sitting in one cannot be told
apart from a sentence re-checked yesterday. Here, every claim carries a date and
a toolchain, so its staleness is visible.

**Toolchain for every claim below unless a section says otherwise:** CUDA 13.0
(nvcc), clang 20.1.8 with libc++, ROCm 7.2.4, AMD target `gfx1200`, NVIDIA target
`sm_86`, reference LAPACK 3.12.0 (Ubuntu 24.04). Dated 2026-09-27.

A section is not re-verified when the toolchain moves. If you upgrade and a claim
here matters to you, re-check it and update the date.

Sections are cited from code by number, so **append rather than insert**.

## Where the vendor facts live

**Anything true of CUDA vs HIP rather than of this project belongs to WarpWraps, and
is written down there:** `deps/WarpWraps/docs/architecture.md` — the wavefront-width
divergence, headers that poison macros, enumerator values that differ where names
agree, `static inline` vendor functions that cannot be re-exported, the
const-correctness divergences between cuBLAS and hipBLAS, and the rest. It is a
vendored submodule, so that file is in this checkout; read it there rather than
restating any of it here. A *second* copy of a vendor fact is how the two rot
apart.

What belongs in *this* file is narrower: numerical and algorithmic decisions, and
whatever the two vendor solver libraries disagree about in a way that reaches this
project's own API.

---

## 1. The backend split is the dependency's, not ours

**Decision.** `calaman_G` has no `src/cuda` and no `src/hip`. Every module under
`src/` is written once against WarpWraps's backend-neutral `wwr*` names and compiles
for either vendor; `CALAMAN_GPU_BACKEND` chooses which, and
`deps/CMakeLists.txt` forwards it as `WWR_GPU_BACKEND`.

**Context.** The alternative — a per-vendor subtree here — duplicates the
problem WarpWraps exists to solve, and doubles it: a routine implemented twice
diverges numerically, and a numerical divergence between backends is far harder
to notice than a compile error.

**Consequences.**

- A `cu*` or `hip*` spelling anywhere in `src/` is a defect unless it sits behind
  a switch that gives *both* backends an answer. Reaching into `wwr.cuda.*` is a
  deliberate act, not a shortcut.
- The cheap check on this is `devtools/cross-backend-check.sh`: it compiles the
  other backend, and nvcc is the more permissive of the two front ends, so
  CUDA-green says little about HIP.
- Anything that cannot be made neutral has to be stated here as a section of its
  own, with what each backend does.

## 2. WarpWraps is consumed with `add_subdirectory`, and what that costs

**Decision.** A git submodule at `deps/WarpWraps`, added as a subdirectory, rather
than a fetched tag or an installed package. One build, one toolchain, one backend
variable, and a two-repo change can be made and tested in one tree.

**Consequences, all of them live today** (`deps/CMakeLists.txt` carries the
detail):

- WarpWraps's CMakeLists adds its own `example/` and `test/` unconditionally — no
  `PROJECT_IS_TOP_LEVEL` guard — so its compile-time tier and examples build as
  part of this project. `WWR_COMPILE_TIME_ONLY=ON` is forwarded to keep its
  *runtime* suites and its GoogleTest fetch out. **The clean fix belongs
  upstream**, and once WarpWraps guards those directories this forwarding can go.
- Vendor packages are found in *this* project's top-level `CMakeLists.txt`,
  because an `IMPORTED` target is visible only in the directory that found it and
  below. Without that, a target reached transitively through a WarpWraps module
  fails to resolve at generate time.
- **The install tier is unresolved.** A calaman target's
  `INTERFACE_LINK_LIBRARIES` names WarpWraps targets, and `install(EXPORT)` refuses
  an export set whose interface names a target no package exports. Either install
  WarpWraps alongside (which needs `wwr::` ALIAS targets in-tree, the spelling an
  installed consumer resolves) or consume an installed WarpWraps through
  `find_package` for install builds. `CALAMAN_INSTALL` is OFF until that is
  decided, and `devtools/install-check.sh` has nothing to prove meanwhile.

## 3. The oracle is the reference LAPACK, and it is deterministic on purpose

**Decision.** Netlib reference LAPACK 3.12.0 plus LAPACKE (`liblapack-dev`,
`liblapacke-dev`), installed in `docker/Dockerfile.base` and exposed to tests as
`calaman::lapack_reference`. Not OpenBLAS, not MKL.

**Context.** A test that says "the GPU factorisation agrees with LAPACK to a
tolerance" is only meaningful if the right-hand side is the same number every
time. A threaded BLAS is not bitwise reproducible across thread counts, so a
tolerance failure would depend on the core count of whoever ran it — the worst
kind of flake, because it reproduces on one machine and not another.

**Consequences.**

- Nothing under `src/` may link the reference. A GPU library that silently falls
  back to a CPU LAPACK is a different library, and the tests could no longer tell
  the two apart.
- Absence is a configure-time `WARNING` and a missing target, so a test that needs
  the oracle must be conditioned on `calaman::lapack_reference` existing. An
  unconditional test would *pass* by not comparing anything, which is the failure
  mode this arrangement exists to prevent.
- Benchmarking against a *fast* CPU BLAS is a separate job from the oracle. Adding
  `libopenblas-dev` repoints Debian's `liblapack.so` alternative and silently
  changes what the oracle is; `docker/Dockerfile.base` has the note.

## 4. LAPACK's names are kept, and its interface is not

**Decision.** A routine that computes what a LAPACK routine computes carries
LAPACK's name (`getrf`, `potrf`, `geqrf`, `gesvd`), and the `s/d/c/z` variants are
one template over the element type rather than four functions.
`.clang-tidy` exempts those names from the project's `lower_case` function rule
for exactly this reason.

**Context.** The name is the one piece of documentation every user of this
library already has. What is *not* kept is the calling convention: no `lwork`
query-then-allocate dance exposed to the caller, no `info` out-parameter as the
only error channel, no leading-dimension argument that has to agree with a
separately passed extent. Those are Fortran-77 accommodations, and WarpWraps's
extension layer already offers handles, device buffers and an error policy.

**Consequence.** For each routine, the mapping from LAPACK's signature to this
project's is a fact worth writing down once, in that module's header — including
which LAPACK behaviours are deliberately *not* reproduced.

## 5. A routine's result structs earn their place one at a time

**Provisional** — drawn from only two routines so far (`calaman.expm`,
`calaman.feast`) and expected to be refined as more modules land. Revisit it
rather than treat it as settled.

**Decision.** The three roles a routine might expose as structs — tuning IN, a
cost *prediction*, an outcome *report* OUT — are each worth a struct **only when
it carries data the others cannot derive**; otherwise they collapse. `expm`
carries a single `ExpmPlan {m, s, num_gemms}`: `expm_plan()` predicts it from the
1-norm and `expm()` reports the one it executed. `feast` keeps a separate
`FeastOptions` and `FeastInfo` and no prediction struct. Both are correct under
the same rule.

**Context.** The reflex is a struct per role, which is how `expm` briefly had an
`ExpmOptions`, an `ExpmPlan` and an `ExpmInfo`. A struct earns its name only by
the fields the others lack:

- **Options** — only if real tuning knobs exist. `expm`'s `ExpmOptions` held one
  enum (`balance`); when balancing moved out of the routine (it is the orthogonal
  similarity `D exp(D^-1 A D) D^-1`, which composes *around* `expm`), the struct
  had nothing left and was deleted. `feast`'s `FeastOptions` (`max_iterations`, `tol`)
  is genuine.
- **A prediction struct** — only where a cost-prediction entry point exists
  (`expm_plan`). Where the routine is also *deterministic*, that same struct is
  the honest report: the degree and squarings follow from the 1-norm alone, so
  what `expm` *will* do equals what it *did*, and a separate `Info` would only
  restate a subset of the plan.
- **An outcome report** — only when the routine yields facts no prediction could
  carry. `FeastInfo` (`iterations`, `max_residual`, `reason`) qualifies: FEAST is
  iterative and data-dependent, so its outcome is not a function of its inputs,
  and it has no predictor — the opposite of `expm` on both counts.

**Consequence.** Before adding `XOptions` / `XPlan` / `XInfo` to a new routine,
ask which fields each holds that the others cannot, and merge or drop the ones
that fail. A deterministic routine with a predictor needs one struct, not three;
an iterative, knob-taking one may well need two.

## 6. Typed constants extend to complex by brace-initialisation

**Decision.** `common/constants.h`'s `kZero … kPi` specialise to
`wwrFloatComplex` / `wwrDoubleComplex` by brace-initialisation (`{re, 0}`), not
through WarpWraps' `make_wwr*Complex`. This is the only place in `src/` that
builds a complex value off the `wwr*` neutral surface, and it is deliberate:
runtime construction still uses the builders (`common/elem_ops.cuh`'s `from_real`
/ `make_complex`), and `elem_ops<T>::zero()/one()` now return these constants
rather than carrying their own `make_##CT(0,0)`, so the zero/one values live in
one place.

**Context.** A `constexpr` variable needs a constant initialiser, and
`make_wwr*Complex` is `inline` (host) / `__device__ __forceinline__` (device),
not `constexpr`, so it cannot initialise one on either backend.
Brace-initialisation can: `cuFloatComplex` is an aggregate `float2`, and
`hipFloatComplex` is a class (`HIP_vector_type<float, 2>`) carrying a `constexpr`
constructor. Measured on the toolchain above, 2026-10-05: `constexpr T z = {0,0}`
compiles for both complex types under CUDA 13.0 and ROCm 7.2.4, while `constexpr
T z = make_wwr*Complex(0,0)` fails on both with "non-constexpr function … cannot
be used in a constant expression". The vendor type shapes themselves are
WarpWraps' domain (`deps/WarpWraps/docs/architecture.md`).

**Consequence.** `constants.h` depends on both vendor complex types being
brace-constructible with `{re, im}` — a property WarpWraps does not promise
through a `constexpr` neutral builder; a future SDK dropping
`HIP_vector_type`'s `constexpr` constructor would break it, and the fallback is
the runtime `elem_ops` builders. Because the specialisations are `inline
constexpr`, they must be present in every TU that instantiates `kZero<complex>`,
so `constants.h` includes `complex.h` unconditionally — which is why the
`:constants` partition now carries the `wwr.device` / `wwr_backend` dependency
`:fp_types` already does.

## 7. The compiler cache is sccache, because ccache cannot cache a module

**Measured 2026-10-05**, clang 20.1.8 + libc++, ccache 4.9.1 vs sccache 0.18.0,
on the shape CMake actually emits for a `.cppm`:

```
clang++ -std=gnu++23 -stdlib=libc++ @<...>.modmap -c interface.cppm -o interface.cppm.o
```

The modmap is a response file carrying `-x c++-module`, one `-fmodule-file=` per
imported module, and `-fmodule-output=`, so **one invocation writes two
outputs**: the object and the BMI.

| | ccache 4.9.1 | sccache 0.18.0 |
|---|---|---|
| that invocation | `Uncacheable calls: 2/2 (100%)` | cold miss, then a **hit** — with both outputs deleted first, the `.pcm` was restored at its full size |
| `clang++ --precompile m.cppm -o m.pcm` | uncacheable | uncacheable (`Non-compilation calls`) |
| a consumer TU (`-fmodule-file=`), a plain `.cpp` | hits | hits |

`src/` is 89 `.cppm` against 38 `.cu`, and `test/` is 65 `.cpp`, so a cache that
refuses the module-interface compile refuses the majority of this tree's
expensive work. That is the whole reason for the choice; nothing else about
ccache was wrong.

**What this replaced.** `Dockerfile.base` installed `ccache` and set
`CCACHE_DIR`, and nothing else: no `CMAKE_<LANG>_COMPILER_LAUNCHER` anywhere, no
`/usr/lib/ccache` on `PATH`, and no `.ccache` bind or volume in any
`devcontainer.json` or in `compose.yaml`. It cached nothing, in any front end,
ever — while `devbox/SKILL.md` and `config.sh` both described it as what made
a second build fast.

**Why `find_program`, not a preset.** The launcher is wired in `CMakeLists.txt`
behind `CALAMAN_COMPILER_CACHE` (default ON) and degrades to a plain compile
when `sccache` is absent. CI's hosted runners have no sccache, and a launcher pinned
in the `base` preset would fail them at the first compile rather than simply not
caching.

**Why `SCCACHE_BASEDIRS` appears nowhere, and must not.** It looks like the fix
for the one real limitation below, and it is a trap twice over. It is read once
when the sccache daemon spawns and is sticky for that daemon's whole life, so
whichever checkout did not spawn the daemon silently gets no benefit. And for
modules it is actively wrong: a restored BMI keeps the absolute path it was
built under, so normalising keys across checkouts mounted at different paths
hands one tree another's BMI and buys a broken build, not a hit.

**The limitation, stated plainly.** One store is shared by every checkout — it
is the primary checkout's `.sccache`, bound into each container at the image's
fixed `SCCACHE_DIR` (`/home/ubuntu/.sccache`) so it is reachable however the
workspace is mounted. But **hits** only cross trees mounted at the *same*
absolute path:

| front end | workspace mount | store | hits |
|---|---|---|---|
| `docker/compose.yaml`, any worktree | always `/workspace` | shared | **shared** |
| devcontainer, main checkout | its own host path | shared | within main |
| devcontainer, `.claude/worktrees/<name>` | its own host path | shared | not with main |

The last row is correct behaviour rather than a misconfiguration: a miss there
is the cache declining to reuse a BMI that would not validate. Sharing the store
still dedups everything each tree rebuilds on its own.

## 8. One `tolerance`, two meanings: relative for lanczos and davidson, backward error for feast

**Decided 2026-10-07** (#240), toolchain as above.

**Decision.** lanczos and davidson share one convergence predicate,
`calaman::classify_ritz`: a Ritz pair converges when its residual is at or below
`tol * max(|theta|, scale)`, with `scale` the 2-norm of the solver's projected
matrix — `max |theta|` over the current subspace spectrum, which both already
hold after their `syevd`/`sygvd`. feast keeps its normwise backward error. There
is no scale-free form of `classify_ritz`.

| Solver | Option | Residual `r` | Converged when |
|---|---|---|---|
| `lanczos` | `LanczosOptions::tolerance` | the estimate `\|beta_m s_{m,i}\|`; with `verify_residuals`, also `\|\|A x - theta x\|\|_2` | `r <= tol * max(\|theta\|, \|\|T\|\|_2)` |
| `davidson` | `DavidsonOptions::residual_tolerance` | `\|\|A x - theta x\|\|_2`; on the metric path the M-norm `sqrt(r^T M r)` | `r <= tol * max(\|theta\|, \|\|H\|\|_2)` |
| `feast` | `FeastOptions::tol` | `\|\|A x - lambda x\|\|_1` | `r / ((\|\|A\|\|_1 + \|lambda\|) \|\|x\|\|_1) < tol`, and the count `m` repeats |

**Context.** Before this, davidson's test was absolute (`r <= tol`) while
lanczos computed the same `||A x - theta x||_2` and compared it to the relative
bound, so the same `1e-8` meant different accuracy in the two, and nothing on a
badly scaled operator. A backward error everywhere was the alternative: davidson
cannot compute one, because it knows the operator only through its `sigma`
callback and never has `||A||_1` — it would need a caller-supplied norm
estimate, a real API addition. feast's backward error is already scale-free and
is reduced on device in the 1-norm; routing it through the host-side
`classify_ritz` would add a sync per iteration for no change in meaning.

**Consequences.**

- The same `tol` asks lanczos and davidson for the same relative accuracy, and
  feast's `tol` is not interchangeable with it. Ritz values lie inside `A`'s
  spectrum, so the relative bound is at most `tol * ||A||_2`; the davidson suite
  asserts exactly that.
- davidson's effective threshold moved from `tol` to
  `tol * max(|theta|, ||H||_2)`: looser on a spectrum larger than 1 in
  magnitude, tighter on a smaller one. On its own suite's problems (`||H||_2`
  between 3.9 and 34 at convergence), iterations fell by one in three of four
  converged cases, converged residuals rose to the new bound, and the
  eigenvalue errors against `LAPACKE_?syevd`/`?sygvd` stayed where they were
  (#240's PR has the per-case numbers, both cards).
- feast compares strictly (`<`), the shared predicate inclusively (`<=`); the
  difference only matters at equality and is left as it was.
