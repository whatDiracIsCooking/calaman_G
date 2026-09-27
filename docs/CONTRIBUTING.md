# Contributing

**This file is pointers, deliberately.** Every fact about how this tree works is
written down exactly once somewhere else, and a CONTRIBUTING.md that restates
"which tier verifies what" becomes a third copy that goes stale silently. What
follows is the map and the short list of things to do before opening a PR.

| Where | What it holds |
|---|---|
| `README.md` | What the library is, and how to use it from a consumer. |
| `CLAUDE.md` | How to work *in* this repo: the containers, the presets, the tiers, and the reasoning behind each. The long answer to almost any question here. |
| `docs/architecture.md` | Why the layers are shaped the way they are. |
| `docs/gpu-header-dependencies.md` | Which vendor headers pull in which. |
| `devtools/config.sh` | Every project-specific setting. Edit this, not the scripts. |

## Setup

```bash
uv sync                           # creates .venv from uv.lock
pre-commit install                # commit-time lint + the pre-push gate
devtools/devcontainer.sh rebuild  # the C++ toolchain lives in the container
```

`uv` is the only assumed host tool. The C++ toolchain — clang-20 with libc++'s
module manifest, CMake 4.2, and a vendor SDK — is not on your host and is not
meant to be; `docker/` builds it.

Two traps, both of which have bitten:

- **Run `pre-commit install` from whichever side you actually commit on.** The
  hook bakes in an absolute Python path and lives in the *common* git dir,
  shared by the main checkout and every worktree. Installing it inside the
  container points it at `/opt/venv` and breaks host commits with
  `ModuleNotFoundError: No module named 'pre_commit'`. Fix with
  `pre-commit install -f` from the right side.
- **After editing `pyproject.toml`, run `uv lock` and commit the result.** The
  lock is tracked and is the only place versions are written; CI and the image
  both install it with `--frozen` and fail rather than re-resolve.

When anything behaves oddly, run **`devtools/doctor.sh`** first — and run it on
both sides, because it is what tells you which half you are on.

## Before you open a PR

CI gates compile, link, export and every non-device test on **both** backends.
What it cannot do is run a kernel, so these are the local gates that matter:

```bash
devtools/cpp-tier.sh              # the full tier. The ONLY thing that runs the
                                  # seven gpu-labelled suites — needs a card.
devtools/cross-backend-check.sh   # ~7s: does the OTHER backend still compile?
devtools/install-check.sh         # install, then build example/consumer
pytest -n auto -rs                # the Python tier (three files, ~2s)
```

- **Anything touching device behaviour needs `devtools/cpp-tier.sh` on a machine
  with a GPU.** Nothing on a hosted runner will catch it for you.
- **A green run means nothing until you know what skipped.** `-rs` prints skip
  reasons; say what skipped and say whether the C++ tier ran at all.
- **`pytest` tests three checker scripts and touches no built binary.** A change
  under `src/` is verified by the C++ tier and by nothing else. Say which suite
  you ran.
- To reproduce a CI result exactly, run CI's own presets:
  `devtools/cpp-tier.sh --preset ci-cuda` / `--preset ci-hip`. Both work with no
  device attached.

## Conventions

C++23, named modules, clang + libc++ (`CMakeLists.txt` refuses gcc); targets are
declared through the macros in `cmake/`, never by hand. `#include` style tracks
ownership: quotes for a header this project owns, angle brackets for the
standard library and vendor SDKs, and **never** a `../` relative climb.
CLAUDE.md's "Conventions" section is the full list, and `docs/` plus the
`docstyle` skill cover how much comment a given file gets.

There is **no formatter hook** — not ruff-format, not clang-format, not
cmake-format, though all three are configured. A formatter rewrites whole files
and buries the real diff in restyling. Lint is narrow (`ruff` on `E,F,I,UP,B`,
plus `cmake-lint`) and must stay clean.

## Pull requests

Everything ships through a PR; `main` is not committed to directly. The template
at `.github/pull_request_template.md` asks which tiers you ran — answer it
honestly, since the device suites are on trust. PRs are squash-merged, so the PR
title becomes the commit subject: plain imperative, no Conventional Commits
prefix.

One caveat with a sharp edge: **a PR that edits `docker/` is tested against the
image `main` already published.** The Dockerfiles are built (not pushed) on such
a PR, so a broken one fails there — but `container:` is resolved before any step
runs, so the `cpp` legs still use the old image. Run the `images` workflow on
your branch first (Actions → images → Run workflow) if the change needs to be
tested against itself.
