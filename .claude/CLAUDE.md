# CLAUDE.md

## What this is

`gpumod` — C++23 module wrappers for the CUDA and HIP GPU APIs, plus the
type-safe abstractions built on them. The vendor headers are exposed as
importable named modules (`import wwr.cuda.cublas_v2;`), the `gpu*` layer
directly under `src/` maps backend-neutral `gpu*` names onto whichever backend
was selected, and `src/wrappers` is written once against those names.

**A build targets exactly one backend.** `WWR_GPU_BACKEND` is `CUDA` or
`HIP`, and it is read *before* `project()` because it decides whether the CUDA
language is enabled at all. A HIP build needs no CUDA toolkit; a CUDA build
needs no ROCm.

The project is being renamed to **Warp Wraps** (`wwr`). The C++ and CMake
identity is already `wwr`: namespace `wwr`, modules `wwr.*`, macros and CMake
options `WWR_*`, CMake helpers `wwr_*`, CMake targets `wwr.*` aliased to
`wwr::*`, and the installed package (`find_package(wwr)`). Still `gpumod`: the
repo, the Python project, and the docker/devcontainer naming below.
`PROJECT_NAME` in `devtools/config.sh` is `gpumod` too — it names docker
volumes, images and the devcontainer, and `doctor.sh` warns when it and any
`.devcontainer/*/devcontainer.json` disagree.

## Setup

```bash
uv sync                                   # creates .venv from uv.lock
pre-commit install                        # commit-time lint + pre-push gate
devtools/devcontainer.sh rebuild          # the C++ toolchain lives in here
```

`uv` is the only assumed host tool, and `uv.lock` is the only place Python
dependency versions live — after editing `pyproject.toml`, `uv lock` and commit
the result; never `uv pip install`. The C++ side's only dependency is
GoogleTest, fetched and built by `deps/CMakeLists.txt` at configure time.

Run `devtools/doctor.sh` first when anything behaves oddly — on a bare host the
whole C++ toolchain warns, which is the expected healthy state. The **doctor**
skill turns each finding into its fix.

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
| [doctor](skills/doctor/SKILL.md) | diagnose a degraded environment — `doctor.sh` plus the failures it cannot see (a stale container, orphaned pytest workers, a stale CMake cache). |
| [docstyle](skills/docstyle/SKILL.md) | write or review source documentation under `src/`/`cmake/` — what belongs in a header vs a declaration vs `docs/architecture.md` vs a README, and the line budget each gets. |
| [audit](skills/audit/SKILL.md) | verify that documentation (skills, READMEs, this file, memory) still matches reality — extract each claim, check it against the tree, report drift. |
| [milestone](skills/milestone/SKILL.md) | turn a plan into a GitHub milestone plus a DAG of PR-sized issues wired for parallel work. |

## Containers: four files, one diamond

```
                Dockerfile.base
                 /           \
  Dockerfile.cuda             Dockerfile.hip
            |                       :
  Dockerfile.combined ..............:  (reuses install-rocm.sh, not the image)
```

| File | What it is |
|---|---|
| `docker/Dockerfile.base` | The vendor-neutral toolchain: clang-20 + libc++, CMake 4.2, Ninja, ccache, uv/Python. No GPU SDK. |
| `docker/Dockerfile.cuda` | `base` + the CUDA toolkit. **The default backend**, and what the devcontainer and compose build. |
| `docker/Dockerfile.hip` | `base` + ROCm. No CUDA at all. |
| `docker/Dockerfile.combined` | `cuda` + ROCm (~40GB). |

The files chain by **tag**, not by stage — each child opens `FROM
${PARENT_IMAGE}` — so **build them only with `docker/build.sh
<base|cuda|hip|combined>`**, which walks the chain and tags the CUDA image both
`:cuda` and `:latest`. Build a child by hand with no parent tagged and docker
tries to *pull* it and fails with `pull access denied`. `combined` is a diamond
only in intent: docker has no multiple inheritance, so it takes `cuda` as its
parent and re-runs `docker/install-rocm.sh` (which is why `ROCM_VERSION` and
`GPU_TARGETS` are declared in both `Dockerfile.hip` and `Dockerfile.combined` —
bump them together).

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
  **compile-and-link plus a thin runtime slice**, not a test of GPU behaviour.
- **What CI still cannot do: the device-dependent suites**
  (`test/extension/{memory_buffer,runtime,rand,blas,solver,fft,sparse}`, the
  `gpu` ctest label). Only a box with a card runs those, so
  **`devtools/cpp-tier.sh` before opening a PR remains the gate for anything
  touching device behaviour** — it is local and bypassable, so run it and say
  what you ran.

`.github/workflows/ci.yml`'s header has the job-by-job breakdown and the
precise reading of what each surviving ctest entry proves; the **pr** skill has
the merge flow. The local tier is deliberately not a git hook — a module build
costs minutes, and a gate that costs minutes gets bypassed; CI carries that
cost instead.

## Conventions

- C++23, named modules, clang + libc++. `CMakeLists.txt` **refuses gcc**.
- Targets are declared through the macros in `cmake/`:
  `wwr_add_cxx_module_library`, the `wwr_add_gtest_*` /
  `wwr_add_test_executable` test macros, and `wwr_add_gpu_device_library`
  for a module's device-kernel `.cu` library. `wwr_add_interface_library` is
  wired and documented with **no call sites** — do not assume it is dead.
- Target names use dots and are aliased to `::`.
- **`#include` style tracks header ownership.** A header this project owns uses
  quotes, spelled by the path its include root makes resolve — bare for a
  same-directory header, root-relative otherwise (`#include
  "wrappers/common/dispatch_sdcz.h"`), **never a `../` relative climb**. The
  standard library and vendor headers use angle brackets. A non-module header a
  `.cppm` includes from its global module fragment must have its include root
  exported (PUBLIC/INTERFACE), not PRIVATE — a PRIVATE root is the export
  regression the **test** skill's package tier exists to catch.
- Python ≥3.13, `from __future__ import annotations` everywhere.
- Lint is deliberately narrow (`E,F,I,UP,B`) with **no formatter hook**. `ruff
  check .` is clean — the pre-commit hook fails on any finding in a file you
  touch. `src/` and `deps/` are excluded (C++); `test/shared/dispatch.py` is
  not.
- **`protect-main.py` guards the primary (`main`) checkout** and is enabled by
  default. Set `CLAUDE_ALLOW_MAIN_EDITS=1` for a session deliberately editing
  it; the guard's own docstring has the full rationale.
- **Install the pre-commit hook from whichever side you commit on.** It bakes an
  absolute `INSTALL_PYTHON` into one file shared by every worktree; installed
  from the wrong side, commits fail with `No module named 'pre_commit'`. The
  **doctor** skill has the fix.
