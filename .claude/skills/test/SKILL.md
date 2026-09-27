---
name: test
description: >-
  Run this repo's suites the right way and interpret the result honestly — the
  C++ tier (devtools/cpp-tier.sh, configure + build + ctest), the Python tier
  (pytest, two files), the package tier (devtools/install-check.sh, install
  then consume), and the cross-backend compile check
  (devtools/cross-backend-check.sh, does the OTHER GPU backend still build) —
  use the right worker count for host vs container, and ALWAYS audit what
  skipped before calling a run green.
  Use when the user asks to run tests, check a change, or verify an edit.
---

# Running tests, and trusting the result

The one rule that matters: **a green run is meaningless until you read the skip
list.** Tests that need a tool the environment lacks skip *silently*, so "all
passed" can mean "the tests that would have caught this never ran." Always pass
`-rs` and read the reasons. On a bare host that is the *normal* case here — the
whole C++ toolchain lives in the CUDA image.

## Two suites, and neither covers the other

| | the C++ suite | the Python suite |
|---|---|---|
| Where | `test/` | `test/shared/` and `.claude/hooks/` (the only pytest files) |
| Language | C++23, googletest | Python, pytest |
| Runner | `ctest` via `devtools/cpp-tier.sh` | `pytest` |
| Tests | the library's own units, including CUDA kernels | two checkers: `test/shared/dispatch.py` and `.claude/hooks/protect-main.py` — no built binary |
| Needs | the CUDA image (clang-20, CMake 4.2, nvcc) | a venv |
| Gated by | nothing automatic — see below | the pre-push hook, only on a `.py` |

There is **no top-level `tests/` directory** — each test lives next to the
script it tests, and `testpaths` names `test/shared` and `.claude/hooks`.

**Which one a change needs is not negotiable by convenience.** A change under
`src/`, `deps/`, `cmake/` or `CMakeLists.txt` is verified by the C++ tier and by
nothing else; a green `pytest` on it means only that the Python that would have
exercised the old binary still imports. Say which suite you ran.

There is also a third tier that is not a suite: `devtools/install-check.sh`
checks the *installed package*, which neither suite touches. See "The package
tier" below for when a change needs it.

And a fourth thing that is not a tier at all: `devtools/cross-backend-check.sh`
compiles the tree for the backend this build is **not** targeting. Every tier
above builds exactly one backend, so none of them can see a portability break.
See "The cross-backend check" below.

## The C++ tier

```bash
devtools/cpp-tier.sh                       # configure + build + ctest, default preset
devtools/cpp-tier.sh --preset debug        # switches the ctest preset too
devtools/cpp-tier.sh --preset asan
devtools/cpp-tier.sh --fresh               # wipe the CMake cache and reconfigure
devtools/cpp-tier.sh --no-test             # configure + build only
devtools/cpp-tier.sh --clean               # rebuild every object, keep the cache
devtools/cpp-tier.sh --tidy                # + clang-tidy over src/*.cpp, ADVISORY
devtools/cpp-tier.sh -j 8                  # parallel compile jobs (BUILD_JOBS)
devtools/cpp-tier.sh -- -R Constants       # everything after -- goes to ctest
devtools/cpp-tier.sh -- -R MemoryBuffer    # ctest -R matches gtest suite names
```

`--tidy` runs `clang-tidy -p <the preset's own build dir>` over the tracked
`src/*.cpp` after a successful build. It is **advisory and never changes the
exit status** — the `.cppm` module interfaces are not clean, and what remains
there is the CRTP design rather than defects. It prints a skip line instead of
failing when clang-tidy is absent.

Run it **inside the container**, which is where the toolchain is:

```bash
devtools/devcontainer.sh shell -c devtools/cpp-tier.sh
```

On a host with no `cmake`/`ninja` the script refuses with exactly that command
rather than half-running — take the refusal at face value; it is not a bug to
work around.

Things that will bite:

- Only `default`, `workstation`, `debug`, `asan`, `hip`, `compile-time`,
  `coverage`, `ci-cuda` and `ci-hip` have **test** presets. With any other
  configure preset the script builds and then reports that there is nothing to
  ctest — which is not the same as passing.
- `default` and `workstation` are the **same configuration**, differing only in
  `binaryDir` (`build/` vs `build-workstation/`), so a container build and a
  host build can coexist instead of reconfiguring each other. Neither expects
  anything prebuilt: the only C++ dependency is GoogleTest, fetched and built
  by `deps/CMakeLists.txt` at configure time. Use `workstation` on a bare host
  purely to keep `build/` — which compose and CLAUDE.md both name — for the
  container.
- `CMAKE_CUDA_ARCHITECTURES=native` queries a live device **at configure time**.
  No GPU, no configure — even for a CPU-only change. `ci-cuda` pins sm_86 and
  is the preset to reach for on a box with no card; there are no other
  per-architecture presets (`volta`/`ampere`/`hopper`/`portable` were removed).
- Every run tees to `.slow-tier-reports/cpp-<stamp>-<rev>-<preset>.log` and
  appends a PASS/FAIL line to `summary.log`. Quote the log path when reporting a
  failure; it has the compiler diagnostics that the summary does not.
- **ctest reports one entry per gtest SUITE**, registered by
  `wwr_add_gtest_suite_tests()` as an `add_test` with a `--gtest_filter`,
  plus a `<target>.SuiteListIsComplete` drift guard per binary. Nothing uses
  `gtest_discover_tests` or `DISCOVERY_MODE PRE_TEST`, so `-R` selects by suite
  name and there is no test-time discovery step.
- **A GPU-less box FAILS the device suites; it does not skip them.** Nothing in
  `test/` calls `GTEST_SKIP` or gates on a device count. The way to run without
  a card is the `gpu` ctest label — the seven `test/extension/*` device targets
  carry it, and `ctest -LE gpu` (which the `ci-cuda` and `ci-hip` test presets
  do) excludes them BY NAME in the output. Prefer that to a skip precisely
  because an exclusion is visible where a skip blends into green.
  `cuda_compile_tests` carries it too by default, for the unrelated reason
  that it links the driver stubs — unless `WWR_CUDA_DRIVER_STUBS=ON`, which
  the `ci-cuda` preset sets so it runs on a driverless runner. That is why
  both CI legs report 12.
  Deliberately no case counts quoted here: the suite grows, and a stale number
  in a skill is worse than none.

## The Python tier: there is only one

**There is one Python tier, not a fast/slow split.** The suite is a few fast
files with no expensive tier behind them, so `FAST_TEST_ARGS` in
`devtools/config.sh` is empty and still prepended to the default run.

```bash
pytest -n auto -rs          # the whole Python suite, under two seconds
pytest -n auto -rs -k name  # narrower
```

Just run all of it; there is no expensive half to hold back. If you reintroduce
a marker in `config.sh`, add something that runs the other half in the same
change — a marker with no runner is a tier that silently never executes.

## The cross-backend check, when the change touches a `.cu`

```bash
devtools/cross-backend-check.sh               # ~7s from the host, docker included
devtools/cross-backend-check.sh --device-only # ~4s, the .cu files only
devtools/cross-backend-check.sh --fresh       # wipe its CMake cache first
```

Run it from the **host** — unlike every other tier, this one is not run inside
the dev container. It does its own one-shot `docker run` against the `hip`
image, because the CUDA container it would otherwise run in has no ROCm at all.

**Run it whenever the change touches a `.cu`, or anything under `src/` a `.cu`
includes.** A `.cu` is compiled by nvcc under CUDA and clang under HIP, and
nvcc is the more permissive of the two — so a whole class of error passes the
C++ tier green and only appears when someone builds ROCm: a functor with a
`const` member of class type, say, which clang rejects (`parallel_for`'s
`device_functor` concept) but nvcc accepts.

Reading the result:

| Exit | Means |
|---|---|
| 0 | the other backend compiles |
| 1 | it does not — the compiler error is in the output |
| 2 | the check could not run (no `hipconfig`, no docker, or no `hip` image) |

**Exit 2 is not a pass.** It prints the `docker/build.sh hip` command it needs
(that script builds `Dockerfile.base` first, which `Dockerfile.hip` requires).
Never report a run that exited 2 as a green cross-backend check.

It proves the other backend **compiles**, not that it runs — no kernel is
launched. `devtools/cpp-tier.sh --preset hip` on a real ROCm box is the
stronger statement; this is the one that costs seconds.

## The package tier, when the change could reach a consumer

```bash
devtools/install-check.sh              # install to a temp prefix, then consume it
devtools/install-check.sh --no-run     # build the consumer, do not run it (no GPU)
devtools/install-check.sh --keep       # keep the prefix to inspect what installed
```

Run it **in the container**, like the C++ tier, and run it whenever the change
touches `cmake/`, a target's usage requirements, or anything under `src/` that a
consumer imports.

**The C++ tier cannot substitute for it.** `cpp-tier.sh` builds the tree in
place and never installs, so no amount of green there says anything about the
installed package. The failure it exists to catch is ordinary rather than
exotic: a consumer of a C++23 module package *compiles the installed module
sources*, so a `.cppm` with a `PRIVATE` include directory or define builds
perfectly here, installs without error, and then fails in every consumer —
`PRIVATE` requirements are not exported. `install-check.sh` installs to a
throwaway prefix and builds `example/consumer` (which reaches gpumod through
`find_package` alone) against it.

Steps 1–3 answer the question; step 4 runs the binary and needs a device. On a
GPU-less box the consumer exits 77 and the script reports a skip, not a failure
— but note the `default` preset still needs a GPU at *configure* time, so pass
`--preset ci-cuda` (it pins sm_86) there, and `--no-run` to stop before step 4
entirely. That is what `.github/workflows/ci.yml`'s `install-check` job runs.

## Host vs container — the worker count differs

- **Host:** `pytest -n auto -rs` from the repo root.
- **Container:** `devtools/devcontainer.sh test` runs the config's `TEST_CMD`
  with `-n $JOBS -rs` inside. `JOBS` defaults to 8 rather
  than `auto` on purpose — several worktree containers each claiming every core
  is how a box ends up thrashing — and every config value honours an
  environment override, so give a concurrent run its own count on the *host*
  side of the call:

  ```bash
  JOBS=4 devtools/devcontainer.sh test        # -> pytest -n 4 -rs
  devtools/devcontainer.sh test -k some_test  # extra pytest args pass through
  ```

  Extra pytest args are appended after the defaults.

  `JOBS` bounds pytest and nothing else. To bound the *container* — so a
  concurrent run in another worktree cannot steal the cores, and a timing
  number means something — pin it with `CPUSET`/`CPUS` on `up`; see the
  `devbox` skill.

For the **Python** suite the container runs exactly what the host runs — it
shells out to fake `llvm-objdump`/`llvm-cxxfilt` stand-ins and needs no device
and no toolchain. The container's advantage is the **C++** tier, which cannot
run on a bare host at all. So a green `pytest` on a laptop is the full Python
result and still says nothing about `src/`.

## The markers: `gpu` and `no_sanitizer`

`pyproject.toml` declares two markers, and **nothing currently uses either** —
the current tests need no device. They are kept as the vocabulary for tests
that do.

- `gpu` — needs a device. These skip on any host without `nvidia-smi`, silently
  unless you passed `-rs`. **This is the marker that would make a laptop run
  look green.**
- `no_sanitizer` — deselected by `docker compose run --rm compute-sanitizer`,
  because instrumentation turns a merely-slow case into an hours-long one. It
  changes nothing about a normal run.

## The batch path

`docker/compose.yaml` runs both suites in one shot, gtest then pytest, logging
to `.log/`:

```bash
export HOST_UID=$(id -u) HOST_GID=$(id -g)
docker compose -f docker/compose.yaml run --rm test
docker compose -f docker/compose.yaml run --rm asan
docker compose -f docker/compose.yaml run --rm compute-sanitizer
```

`SKIP_GTEST=1` / `SKIP_PYTEST=1` narrow it to one suite, `TEST_FILTER` is
a regex for `ctest -R` and `PYTEST_ARGS` is appended to pytest. Prefer this when you
want a clean throwaway run; prefer `cpp-tier.sh` inside the devcontainer when
you are iterating. See the `devbox` skill for the difference.

## After the run: audit the skips, then the workers

1. **Read the `-rs` skip reasons.** Separate the expected ones (a test
   reporting a measurement, a case that does not apply here) from "that
   toolchain is absent and its coverage did not run" — and say the latter
   explicitly rather than claiming the change is verified.
   `devtools/doctor.sh` reports the whole tool list in one command (the
   `doctor` skill turns each of its findings into a fix).

2. **If you interrupted a container run, check for orphaned workers.** Killing
   `devtools/devcontainer.sh test` on the host kills only npx — the pytest
   master and its xdist workers keep running inside at 100% CPU, and nothing
   tells you; the next run just comes out slower and you measure *that*:

   ```bash
   devtools/devcontainer.sh shell -c "ps -eo pid,pcpu,cmd --sort=-pcpu | head"
   ```

   `pkill -f pytest` reaches only the master (a worker's command line is a bare
   `python3 -u -c import sys;...`), so kill the workers **by PID**.

## What to report

Say **which suite** ran, which tier of it, how many passed, and what skipped and
why. "N passed, M skipped — all M because <tool> is absent, so <what that
covers> was not verified here" is an honest result; "all green" on the same run
is not.

And say what did *not* run at all. A pytest-only result on a change under `src/`
should be reported as exactly that: "the Python tier passes; the C++ tier did
not run here (no GPU on this host), so the change is unverified." That sentence
is the difference between a useful report and a misleading one.
