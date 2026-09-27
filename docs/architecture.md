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

**Anything true of CUDA vs HIP rather than of this project belongs to gpumod, and
is written down there:** `deps/gpumod/docs/architecture.md` — the wavefront-width
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
`src/` is written once against gpumod's backend-neutral `wwr*` names and compiles
for either vendor; `CALAMAN_GPU_BACKEND` chooses which, and
`deps/CMakeLists.txt` forwards it as `WWR_GPU_BACKEND`.

**Context.** The alternative — a per-vendor subtree here — duplicates the
problem gpumod exists to solve, and doubles it: a routine implemented twice
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

## 2. gpumod is consumed with `add_subdirectory`, and what that costs

**Decision.** A git submodule at `deps/gpumod`, added as a subdirectory, rather
than a fetched tag or an installed package. One build, one toolchain, one backend
variable, and a two-repo change can be made and tested in one tree.

**Consequences, all of them live today** (`deps/CMakeLists.txt` carries the
detail):

- gpumod's CMakeLists adds its own `example/` and `test/` unconditionally — no
  `PROJECT_IS_TOP_LEVEL` guard — so its compile-time tier and examples build as
  part of this project. `WWR_COMPILE_TIME_ONLY=ON` is forwarded to keep its
  *runtime* suites and its GoogleTest fetch out. **The clean fix belongs
  upstream**, and once gpumod guards those directories this forwarding can go.
- Vendor packages are found in *this* project's top-level `CMakeLists.txt`,
  because an `IMPORTED` target is visible only in the directory that found it and
  below. Without that, a target reached transitively through a gpumod module
  fails to resolve at generate time.
- **The install tier is unresolved.** A calaman target's
  `INTERFACE_LINK_LIBRARIES` names gpumod targets, and `install(EXPORT)` refuses
  an export set whose interface names a target no package exports. Either install
  gpumod alongside (which needs `wwr::` ALIAS targets in-tree, the spelling an
  installed consumer resolves) or consume an installed gpumod through
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
separately passed extent. Those are Fortran-77 accommodations, and gpumod's
extension layer already offers handles, device buffers and an error policy.

**Consequence.** For each routine, the mapping from LAPACK's signature to this
project's is a fact worth writing down once, in that module's header — including
which LAPACK behaviours are deliberately *not* reproduced.
