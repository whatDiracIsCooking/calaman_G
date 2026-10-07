# CLAUDE.md

## What this is

`calaman_G` — GPU-accelerated dense linear algebra and solvers in C++23 named
modules: LAPACK-shaped work (factorisations, linear solves, least squares,
eigenproblems) on the device, with the netlib reference LAPACK on the CPU as the
oracle its tests check against.

**It is agnostic to NVIDIA and AMD, and it owns none of that machinery.** The
backend split lives in the dependency: [WarpWraps](https://github.com/whatDiracIsCooking/WarpWraps), a git submodule at
`deps/WarpWraps`, exposes the vendor headers as importable modules and maps its
backend-neutral `wwr*` names onto whichever backend was selected. So there is
**no `src/cuda` and no `src/hip` here** — `src/` is one tree, written once
against those names, built for either vendor.

**A build targets exactly one backend.** `CALAMAN_GPU_BACKEND` is `CUDA` or
`HIP`, read *before* `project()` because it decides whether the CUDA language is
enabled at all, and forwarded to WarpWraps as `WWR_GPU_BACKEND` from
`deps/CMakeLists.txt`. A HIP build needs no CUDA toolkit; a CUDA build needs no
ROCm.

The C++ and CMake identity is `calaman`: namespace `calaman`, modules
`calaman.*`, C++ preprocessor macros `CLM_*`, CMake options `CALAMAN_*`, CMake
helpers `calaman_*`, CMake targets `calaman.*` aliased to `calaman::*`, and the
installed package would be `find_package(calaman)`. The repo and the GitHub project are
`calaman_G`; docker images, volumes and the devcontainer are `calaman`
(`PROJECT_NAME` in `devtools/config.sh`), because a docker repository name
cannot carry a capital. `doctor.sh` warns when `PROJECT_NAME` and any
`.devcontainer/*/devcontainer.json` disagree.

## The state of the tree — read this before trusting any tier

`src/` and `test/` **are live and GPU-exercised.** `src/` holds **one module per
LAPACK routine, each in its own `src/` directory** — ~40 of them now, spanning the
`calaman.common`/`calaman.diff_norm`/`calaman.lacpy` utilities, pivoted Cholesky
(`pstrf`/`pstf2`), the matrix exponential (`expm` + `paterson_stockmeyer` +
`horner`), balancing (`gebal`/`gebak`), the `feast` eigensolver, `nnls`, and the
Hessenberg/Schur reduction chain (`gehd2`, `lahr2`, `larfb`, `larft`, `lanv2`,
`lartg`, `lasy2`, …). The `geqp3` column-pivoted-QR call graph is the worked
example of the shape — `calaman.larfg`, `calaman.larf`, `calaman.laqp2`,
`calaman.laqps`, and the `calaman.geqp3` driver (all-free and fixed-prefix) — with
the inter-routine edges as real module imports (`laqp2` imports `larfg`+`larf`,
`laqps` imports `larfg`, `geqp3` imports `laqp2`+`laqps`), not partitions of one
umbrella module. The test tier checks each against the reference LAPACK on a real
card. What is still NOT live is the install tier; the table says which is which,
so report the tier you actually touched.

| | State |
|---|---|
| `deps/` | **live.** WarpWraps submodule + GoogleTest fetch. |
| `docker/`, `.devcontainer/`, `devtools/` | **live**, and the LAPACK layer in `docker/Dockerfile.base` is new here. |
| `src/`, `test/` | **live.** The modules above plus their suites; `add_subdirectory(src)` and `(test)` are enabled. `#add_subdirectory(example)` stays commented — `example/` is a `README.md` stub with no `CMakeLists.txt` yet (a directory with no `CMakeLists.txt` is a configure error). |
| `experimental/` | **live but opt-in.** `add_subdirectory(experimental)` is gated behind `CALAMAN_BUILD_EXPERIMENTAL` (default **OFF**) and runs after `src/` so an experimental module may link a shipped one; ships `calaman.experimental.xor_delta` and `calaman.experimental.byte_transpose`. |
| `CMakeLists.txt` | **live**: toolchain discovery, backend choice, vendor packages, the LAPACK oracle, WarpWraps, and now `src/`, `test/`, and the gated `experimental/`. |
| the `cmake/` target macros | **exercised** — `calaman_add_cxx_module_library`, the `calaman_add_gtest_*` macros, and `calaman_add_gpu_device_library` all have call sites now, and so does `calaman_add_interface_library` (`calaman.sym2x2`, the header-only device helpers in `src/lapack/sym2x2/`). |
| `calaman_install.cmake` | dormant, but **no longer wrong-shaped**: it does one recursive sweep of the single `src/` tree now; the old per-backend `src/{cuda,hip,wrappers}` sweep inherited from WarpWraps is gone. |
| the install tier | **dormant.** `CALAMAN_INSTALL` defaults OFF; `calaman_install_package()` is still expected to fail on the unexported `wwr.*` interface targets until the two-package problem (docs/architecture.md §2) is decided. `devtools/install-check.sh` and CI's `install-check` job have nothing to prove until then. |
| the C++ test tiers | **live and asserting.** The numerical suites compare device results against the reference LAPACK to a shared tolerance (`test/shared/tolerance.cppm`); they are `REQUIRES_GPU`, so `ctest -LE gpu` excludes them. One host-only suite (`linalg_scaffold_tests`) asserts without a card. |

When you run something, say which of these it touched.

## Setup

```bash
git submodule update --init --recursive   # deps/WarpWraps
uv sync                                   # creates .venv from uv.lock
pre-commit install                        # commit-time lint + pre-push gate
devtools/devcontainer.sh rebuild          # the C++ toolchain lives in here
```

`uv` is the only assumed host tool, and `uv.lock` is the only place Python
dependency versions live — after editing `pyproject.toml`, `uv lock` and commit
the result; never `uv pip install`. The C++ side has two dependencies and builds
both from source: **WarpWraps** from the submodule, and **GoogleTest**, fetched by
`deps/CMakeLists.txt` at configure time. The one prebuilt dependency is the CPU
reference LAPACK, which is a distro package in the image.

Run `devtools/doctor.sh` first when anything behaves oddly — on a bare host the
whole C++ toolchain warns, which is the expected healthy state. The **doctor**
skill turns each finding into its fix.

## The dependency: what comes from WarpWraps, and what is ours

Everything under `src/` here imports WarpWraps rather than a vendor header
directly. Its layers, outermost first — prefer the outermost one that does the
job:

| WarpWraps layer | What it is | Example |
|---|---|---|
| `wwr.extension.*` | handles, error policies, device buffers, `parallel_for`, convenience calls | `wwr.extension.solver` |
| `wwr.wrappers.*` | type-safe templates over the neutral layer, dispatched on `s/d/c/z` | `wwr.wrappers.solver` |
| `wwr.blas`, `wwr.solver`, … | the backend-neutral layer: one `wwr*` name per vendor name | `wwrsolverDnXgetrf` |
| `wwr.cuda.*` / `wwr.hip.*` | the raw 1:1 vendor header modules | `wwr.cuda.cusolverDn` |

**Reaching for a lower layer is a decision, not a convenience.** A call that
lands in `wwr.cuda.*` is a call that only compiles on one backend, so it needs a
counterpart on the other or it breaks the promise in the first paragraph of this
file.

Two consequences of consuming WarpWraps with `add_subdirectory`, both written up
in `deps/CMakeLists.txt`:

- **Its compile-time tier and examples build with this project** (it has no
  `PROJECT_IS_TOP_LEVEL` guard on those). `CALAMAN_WARPWRAPS_TESTS=OFF` — the
  default — is what keeps its *runtime* suites and its GoogleTest fetch out.
  The clean fix belongs upstream.
- **Vendor packages are found in the top-level `CMakeLists.txt`**, not left to
  WarpWraps, because an IMPORTED target is visible only in the directory that found
  it and below. Keep that list in step with `deps/WarpWraps/CMakeLists.txt`.

Bump the submodule deliberately (`git -C deps/WarpWraps fetch && git -C deps/WarpWraps
checkout <sha>`, then commit the gitlink), and run the C++ tier after — a
dependency bump is exactly the change that a green Python run says nothing about.

## The CPU reference LAPACK

`docker/Dockerfile.base` installs `liblapack-dev` + `liblapacke-dev` +
`gfortran`, and `CMakeLists.txt` exposes them to tests as
`calaman::lapack_reference` (LAPACKE's C interface, `lapacke.h`).

- **Reference, not OpenBLAS, on purpose.** An oracle has to be deterministic,
  and a threaded BLAS is not bitwise reproducible across thread counts.
  `Dockerfile.base` has the `update-alternatives` trap to read before anyone
  adds OpenBLAS for benchmarking.
- **Tests only.** Nothing under `src/` may link it. A GPU library that quietly
  falls back to a CPU LAPACK is a different library, and the tests could no
  longer tell the two apart.
- **Missing is a WARNING, not an error**, so a test that needs the oracle must
  check for the target rather than assume it — otherwise its absence turns into
  a silent pass instead of a missing tier.

## Skills

Most of the how-to for working in this repo lives in `.claude/skills/`, and the
detailed rationale lives in the code it describes — file headers, script
headers, `docs/architecture.md`, and the per-directory `README.md` files. This
file is the map; reach for the skill when you act.

| Skill | Reach for it to… |
|---|---|
| [test](skills/test/SKILL.md) | run the suites the right way and read the result honestly — the C++ tier (`cpp-tier.sh`), the Python tier (`pytest`), the package tier (`install-check.sh`), the cross-backend compile check, and coverage. Presets, `gpu`-label exclusions and what a green run does *not* prove all live here. |
| [devbox](skills/devbox/SKILL.md) | drive the containers and the container-backed **sibling** worktrees — `devcontainer.sh` (up/rebuild/shell/test/down) and `worktree.sh` (add/rm/sync/gc). Rebuild-not-up, CPU bounds, and the docker/compose batch path. |
| [worktree](skills/worktree/SKILL.md) | create and clean up the lightweight `.claude/worktrees/<name>` checkouts — no container, cheap and disposable. Not the sibling worktrees (that is `devbox`). |
| [pr](skills/pr/SKILL.md) | ship the current work end to end: commit → push → PR → merge, with the repo's gotchas (never on `main`, fill the template, never `--admin`/`--delete-branch`). |
| [issue](skills/issue/SKILL.md) | resolve one GitHub issue end to end — `/issue <N>`: fresh worktree, implement, run the relevant suites on **both** cards (CUDA + HIP), then `/pr full` and merge (invoking it is the merge grant for that PR). |
| [doctor](skills/doctor/SKILL.md) | diagnose a degraded environment — `doctor.sh` plus the failures it cannot see (a stale container, orphaned pytest workers, a stale CMake cache). |
| [docstyle](skills/docstyle/SKILL.md) | write or review source documentation under `src/`/`cmake/` — what belongs in a header vs a declaration vs `docs/architecture.md` vs a README, and the line budget each gets. |
| [codestyle](skills/codestyle/SKILL.md) | write or review C++ source conventions nothing enforces — `#include` style first: `"quotes"` for this project's headers, `<angles>` for everything external including WarpWraps, plus a check for the tree; and constant naming (`kCamelCase` at namespace/class scope, `lower_case` allowed for locals, all-caps matrix names like `A`/`d_A`). |
| [workspace](skills/workspace/SKILL.md) | write or review a routine's device workspace — the carve-once convention (`slices_for` + `carve_workspace` + `WorkspaceLayout`), the canonical `XSlices`/`make_X_slices`/`*_bufferSize` shape, fixed-vs-scratch rules, and the review checklist. |
| [audit](skills/audit/SKILL.md) | verify that documentation (skills, READMEs, this file, memory) still matches reality — extract each claim, check it against the tree, report drift. |
| [milestone](skills/milestone/SKILL.md) | turn a plan into a GitHub milestone plus a DAG of PR-sized issues wired for parallel work. |
| [milestone-run](skills/milestone-run/SKILL.md) | execute a milestone — `/milestone-run <url> [max=N] [both] [dry-run]`: fan out up to N subagents, one per ready issue in its own `.claude/worktrees/` checkout, each tested on one card (alternating CUDA/HIP; `both` requires both); the orchestrator merges their PRs one at a time and launches whatever becomes unblocked (invoking it is the merge grant for that milestone's PRs). |

## Containers: four files, one diamond

```
                Dockerfile.base
                 /           \
  Dockerfile.cuda             Dockerfile.hip
            |                       :
  Dockerfile.combined ..............:  (reuses the HIP install scripts, not the image)
```

| File | What it is |
|---|---|
| `docker/Dockerfile.base` | The vendor-neutral toolchain: clang-20 + libc++, CMake 4.2, Ninja, sccache, uv/Python, **and the CPU reference LAPACK**. No GPU SDK. |
| `docker/Dockerfile.cuda` | `base` + the CUDA toolkit, plus the libraries the toolkit does not carry: NCCL, cuTENSOR, nvCOMP (apt) and cuGraph (wheels, `/opt/rapids`). **The default backend**, and what the devcontainer and compose build. |
| `docker/Dockerfile.hip` | `base` + ROCm, plus hipCOMP, which AMD packages nowhere — built from source into `/opt/rocm-ds`. No CUDA at all. |
| `docker/Dockerfile.combined` | `cuda` + ROCm (~40GB). |

The files chain by **tag**, not by stage — each child opens `FROM
${PARENT_IMAGE}` — so **build them only with `docker/build.sh
<base|cuda|hip|combined>`**, which walks the chain and tags the CUDA image both
`:cuda` and `:latest`. Build a child by hand with no parent tagged and docker
tries to *pull* it and fails with `pull access denied`. `combined` is a diamond
only in intent: docker has no multiple inheritance, so it takes `cuda` as its
parent and re-runs `docker/install-rocm.sh` and `docker/install-rocm-ds.sh`
(which is why `ROCM_VERSION`, `GPU_TARGETS` and `HIPCOMP_VERSION` are declared in
both `Dockerfile.hip` and `Dockerfile.combined` — bump them together).

**Claude Code itself is pinned in the three GPU files**, not installed by a
devcontainer feature: each ends with `docker/install-claude-code.sh` (Node 22,
then `@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}`, then a check that what
landed is what was asked for). It is at the leaves rather than in `base` so a
bump re-runs one layer instead of the CUDA toolkit and ~19GB of ROCm, which
means the version is declared **three times** — bump it with
`devtools/claude-version.sh --apply`, which rewrites all three; `doctor.sh` and
`devcontainer.sh up`/`rebuild` report when they disagree or the pin is behind
the registry. The feature it replaced npm-installed an *unpinned* version into
a permanently cached layer, and `devcontainer-lock.json` pins that feature's
digest rather than the version it installs, so images drifted silently.

Two front ends, both onto the `cuda` image: **`docker/compose.yaml`** for
one-shot batch runs, and **`.devcontainer/cuda/`** (via `devtools/devcontainer.sh`)
for interactive work. They cannot share a build directory — a CMake cache
records absolute paths and the two mount the workspace at different ones.

Depth lives where it is used: `docker/README.md` has the variables, the six
images (the `-ci` variants and `ROCM_PRUNE`), and the compose caveats;
`docker/install-rocm.sh` has the prune list; the **devbox** skill drives all of
it.

## What is gated, and what is not

- **`git push`** runs the fast pytest tier — but **only when the push touches a
  `.py`** (the `files: \.py$` filter on the pre-push hook).
- **CI runs on every push to `main` and every PR**, all on GitHub-hosted
  runners with no GPU. A docs-only change skips the heavy jobs; a real code
  change does not. **`ci-ok` is the one aggregate check name stable enough to
  require in a branch ruleset** — every other name is generated and moves.
- Both backends are built on the server and run `ctest -LE gpu` — read that as
  **compile-and-link plus whatever CPU-only tests exist**, not as a test of GPU
  behaviour. Today that is compile-and-link plus the one host-only suite
  (`linalg_scaffold_tests`); every numerical suite is `REQUIRES_GPU` and excluded,
  so CI still proves nothing about a kernel's numbers. A third leg, `cpp
  (asan)` (preset `ci-asan`), runs that same host-only set under ASan+LSan,
  plus the host sanitizer canaries that prove ASan is on; device code is never
  sanitized in CI (`docs/sanitizers.md`, G5).
- **What CI cannot do: run a kernel.** Every numerical claim this project makes
  — a factorisation that agrees with the reference LAPACK to a tolerance — needs
  a card, which no hosted runner has. So **`devtools/cpp-tier.sh` on a box with
  a GPU before opening a PR is the gate for anything touching device
  behaviour** — it is local and bypassable, so run it and say what you ran.
- **A submodule bump is a code change**, and CI treats it as one.

`.github/workflows/ci.yml`'s header has the job-by-job breakdown; the **pr**
skill has the merge flow. The local tier is deliberately not a git hook — a
module build costs minutes, and a gate that costs minutes gets bypassed; CI
carries that cost instead.

## Conventions

- C++23, named modules, clang + libc++. `CMakeLists.txt` **refuses gcc**.
- Targets are declared through the macros in `cmake/`:
  `calaman_add_cxx_module_library`, the `calaman_add_gtest_*` /
  `calaman_add_test_executable` test macros, and
  `calaman_add_gpu_device_library` for a module's device-kernel `.cu` library.
  `calaman_add_interface_library` declares a header-only target (today only
  `calaman.sym2x2`, the shared `__device__` ?lae2/?laev2/?lapy2 ports).
- Target names use dots and are aliased to `::`; link lists name the `::`
  alias, and the installed package exports that same spelling
  (`cmake/README.md`, "One spelling").
- **Backend-neutral by construction.** A `wwr*` name from WarpWraps is the portable
  spelling; a `cu*`/`hip*` name in this tree is a bug unless it sits behind a
  switch that gives both backends an answer.
- **The umbrella target is `calaman_compile_time_tests`**, not
  `compile_time_tests` — WarpWraps defines that second name, and two targets cannot
  share one.
- **`#include` style tracks header ownership.** A header this project owns uses
  quotes, spelled by the path its include root makes resolve — bare for a
  same-directory header, root-relative otherwise — **never a `../` relative
  climb**. Everything external uses angle brackets — the standard library, the
  vendor SDKs, LAPACKE/CBLAS, GoogleTest, **and WarpWraps** (`<runtime.h>`,
  `<complex.h>`, `<wrappers/math/math.cuh>`: a submodule, not ours, even though
  its `-I` root would let quotes compile). A non-module header a `.cppm`
  includes from its global module fragment must have its include root exported
  (PUBLIC/INTERFACE), not PRIVATE. The **codestyle** skill has the table and a
  check.
- **Documentation under `src/`/`cmake/` follows the `docstyle` skill — apply it
  when you write or edit a header, declaration, or CMake comment, not only when
  asked to review.** Its budget (25-line header, 5-line declaration) is enforced
  by nothing at commit or in CI, so an unchecked edit is exactly how headers
  bloat; run the skill's budget check after touching one. The judgement the skill
  encodes — constraint vs story, invariant vs mechanism — only constrains if it is
  exercised at the keystroke.
- Python ≥3.13, `from __future__ import annotations` everywhere.
- Lint is deliberately narrow (`E,F,I,UP,B`) with **no formatter hook**. `ruff
  check .` is clean — the pre-commit hook fails on any finding in a file you
  touch. `src/` and `deps/` are excluded (C++).
- **`protect-main.py` guards the primary (`main`) checkout** and is enabled by
  default. Set `CLAUDE_ALLOW_MAIN_EDITS=1` for a session deliberately editing
  it; the guard's own docstring has the full rationale.
- **Install the pre-commit hook from whichever side you commit on.** It bakes an
  absolute `INSTALL_PYTHON` into one file shared by every worktree; installed
  from the wrong side, commits fail with `No module named 'pre_commit'`. The
  **doctor** skill has the fix.
