---
name: doctor
description: >-
  Diagnose a degraded environment: run devtools/doctor.sh and turn each
  finding into the fix, plus the failures it cannot see (orphaned pytest
  workers, a container that ignored your config edit). Use when something
  behaves oddly — commits fail with "No module named pre_commit", tests skip or
  vanish, git says "not a repository" inside a container, a worktree command
  cannot find its docker state — or when the user just asks what is wrong here.
---

# Reading a degraded environment

```bash
devtools/doctor.sh
```

It exits non-zero **only** on a FAIL (the core suite cannot run). A WARN means
some slice of the suite will skip, which is often the correct state of a
machine. So the output is not pass/fail: it is a list of what this environment
cannot do, and the job is to say which of those matter for the task at hand.

Run it **first** when anything behaves oddly, and again after any fix.

## Run it on both sides, and compare

This repo has a host half and a container half, and doctor is the thing that
tells you which one you are on:

```bash
devtools/doctor.sh                                    # the host
devtools/devcontainer.sh shell -c devtools/doctor.sh  # the CUDA container
```

On a bare host, **around a dozen warnings is the expected, healthy state**.
What is genuinely container-only is the clang/GPU toolchain -- `cmake`,
`ninja`, `clang++`, `clang-scan-deps`, `clang-format`, `llvm-objdump`,
`llvm-cxxfilt`, `nvcc`, `compute-sanitizer`, `nsys`, `ccache` -- and
`hipconfig`, which is present only in the images built from
`docker/Dockerfile.hip` and `docker/Dockerfile.combined`, not the `cuda` one.

Note what is NOT on that list any more: `cmake-format` and `cmake-lint` come
from the `cmakelang[yaml]` dev dependency, and doctor resolves tools through
`VENV_PATHS` (`find_tool` in `devtools/lib.sh`) rather than `PATH`. On the host
they report `[ ok ]` with a `not on PATH; using .../.venv/bin/...` note, which
is normal rather than a finding. That is not a broken machine; it is the
laptop side of a two-sided workflow, good for editing, linting and the Python
suite. The same list appearing *inside* the container is a real problem.

Never report the host list on its own as "the environment is degraded." Say
which side you ran on, and what the other side would cover.

## Finding → what it means → fix

**`[FAIL] not inside a git repository`** — nothing else can be checked. Usually
means a container without the `.git` bind mount: a worktree's `.git` is a file
pointing outside the workspace folder, so `.devcontainer/<variant>/devcontainer.json`
mounts the main checkout's `.git` common dir inside. `devtools/devcontainer.sh`
injects that host path as `WWR_GIT_DIR` on `up`/`rebuild`, so the usual cause
is a container brought up another way (a direct `devcontainer up`, or VS Code
"Reopen in Container" with the fallback path unedited). Bring it up with
`devtools/devcontainer.sh rebuild`; `up` will not apply the mount change.

**`[warn] <file> names volumes that do not start with '<name>-'`** — a
half-applied rename. `PROJECT_NAME` is spelled out in `devtools/config.sh` *and*
in **every** `devcontainer.json` file (JSON cannot source shell), and while they
disagree `worktree.sh rm` and `gc` do not recognise this project's volumes, so
every worktree you tear down leaks its whole set. Each file is checked
separately, so the warning names which one drifted. The fix is to make the odd
one out agree with `PROJECT_NAME` in `devtools/config.sh`.

**`[warn] cannot resolve this repo's git dir for the container .git mount`** —
`doctor` ran somewhere `git rev-parse --path-format=absolute --git-common-dir`
returns nothing, so the path `devcontainer.sh` would inject as `WWR_GIT_DIR`
is empty and the container's `.git` mount would fail. Run doctor from inside the
gpumod checkout.

**`[FAIL] DEVCONTAINER_CONFIG does not exist`** — `devtools/config.sh` points at
a `devcontainer.json` that is not there, and *every* `devcontainer.sh` command
refuses until it is fixed. The shipped values are
`.devcontainer/cuda/devcontainer.json` (the default), `.devcontainer/hip/devcontainer.json`
and `.devcontainer/combined/devcontainer.json`.

**`[warn] worktree root not writable` / `primary checkout owned by uid N`**
— `worktree.sh add` cannot create sibling checkouts, and git ops in main fail.
Fix once with the numeric uid: `sudo chown -R $(id -u) <root>`. Not `chown
ubuntu` — the host user and the container's `ubuntu` are usually both uid 1000
under different names, so the name form fails on the host.

**`[warn] N broken/prunable linked worktree(s)`** — a worktree directory was
deleted out from under git. `git worktree prune` clears the bookkeeping; then
check `devtools/worktree.sh gc --dry-run`, because a hand-deleted worktree
usually left its container and volumes behind too.

**`[warn] no venv found` / `pytest not found` / `uv not found`** — `uv sync`.
`uv` is the only assumed host tool; the venv is built from the tracked
`uv.lock`, never with `uv pip install`.

**`[warn] uv.lock does not name the project '<name>' -- it is stale`** — the
lock and `pyproject.toml` disagree, usually after a dependency edit or a
half-finished rename. `uv sync --frozen` is what both Dockerfiles and CI run, so
this surfaces as a failed **image build** a long way from the cause. Fix with
`uv lock` and commit the result; do not hand-edit the lock, and do not "work
around it" by dropping `--frozen`, which is the line that keeps the container
and the host on the same versions.

**`[warn] <tool> not found -- lost: <what>`** — an optional tool from
`DOCTOR_OPTIONAL_TOOLS` in `devtools/config.sh`. Install it *or* accept that its
slice of the suite will skip — but say which, because that is exactly the
coverage a green run will not have. This list is the project's to maintain: when
a test starts depending on a tool, add it here.

**`[warn] gh cannot see <owner>/<repo>`** — the token authenticates but was
never granted THIS repository. A fine-grained PAT names its repositories one by
one, and `gh auth status` reports a clean login either way, so without this
check the first symptom is `gh pr create` failing with "Could not resolve to a
Repository" *after* the branch is already pushed. Fix it under Repository
access on the token, not by re-authenticating. Blocks the `pr` and `milestone`
skills entirely.

**`[warn] gh present but GH_TOKEN empty`** — `gh` is installed but
unauthenticated, so the PR flow fails at push time rather than at the start.
Export a fine-grained PAT (Contents + Pull requests: read/write) on the **host**
and `rebuild`: the container takes `GH_TOKEN` from the host env at create time,
so an `export` after the container is up does not reach it.

**`[warn] <path> absent`** — a `DOCTOR_REQUIRED_PATHS` entry. A *relative* one
is typically a submodule (`git submodule update --init --recursive`, which
doctor suggests).

`DOCTOR_REQUIRED_PATHS` is currently **empty**, so this warning should not
appear at all. That is deliberate: the C++ tree's only dependency is GoogleTest,
which `deps/CMakeLists.txt` fetches and builds from source at configure time, so
no preset reads anything prebuilt in the image. If this warning does appear,
someone added a path — check that a build actually reads it before installing
anything.

**`[warn] pre-commit hook unusable` / `no pre-commit hook installed`** — the
one footgun worth memorising. The hook bakes an absolute `INSTALL_PYTHON` and
lives in the **common** git dir: one file shared by the main checkout and every
worktree. Installed from inside the container it points at `/opt/venv` and every
host commit then fails with `ModuleNotFoundError: No module named 'pre_commit'`;
installed on the host it breaks committing inside. Fix with `pre-commit install
-f` **from whichever side you commit on**. `postCreateCommand` is guarded so
bringing a container up will not do this to you — do not remove that guard.

**`[warn] N orphaned docker resource(s)`** — containers, volumes and images from
worktrees that no longer exist (a plain `git worktree remove`, or an agent's
`.claude/worktrees/` checkout). Review with `devtools/worktree.sh gc --dry-run`,
then `gc` to remove. The count comes from that same `--dry-run`, so the two
always agree.

## What doctor.sh cannot see

Five failures that look like something else entirely:

1. **A container running your old config.** `up` reuses the running container
   and silently applies no change from `devcontainer.json` or the `Dockerfile`,
   while reporting success and the same container id. If a change "did not take
   effect", `devtools/devcontainer.sh rebuild` before debugging anything else.

2. **Orphaned pytest workers.** Killing `devtools/devcontainer.sh test` on the
   host kills `npx` only; the pytest master and its xdist workers keep running
   inside at 100% CPU. Symptom: every later run is inexplicably slower.

   ```bash
   devtools/devcontainer.sh shell -c "ps -eo pid,pcpu,cmd --sort=-pcpu | head"
   ```

   `pkill -f pytest` reaches only the master — kill the workers by PID.

3. **A green run that tested nothing.** Skips are silent without `-rs`, and a
   suite whose tests all degrade-to-skip on a missing tool passes cleanly. This
   is the failure doctor exists for, but it only tells you the *tools* are
   missing — the `test` skill covers reading the skip list itself.

4. **A stale CMake cache.** Doctor never configures, so it cannot see that
   `build/` holds a toolchain variable you changed an hour ago. A change that
   "does nothing" wants `devtools/cpp-tier.sh --fresh`; a plain reconfigure
   keeps the old value.

5. **A device-behaviour change with no gate behind it.** The push hook only
   fires on `.py`. CI now builds both backends and runs `ctest -LE gpu`, so
   compile, link and the non-device tests *are* checked server-side — but the
   `gpu`-labelled device suites (`test/extension/*`) are excluded, so a change
   that only a card would catch stays ungated until `devtools/cpp-tier.sh` runs
   locally. Doctor reports tools, not that absence.

## Reporting

Say what is degraded and what it costs, not just the counts: "no failures, 21
warnings — all the C++ toolchain, because this is the host and it lives in the
CUDA image; the Python suite runs here, the C++ tier does not." A bare "doctor
passes" throws away the entire point of the command, and a bare "21 warnings"
reads as alarming when it is the expected state.
