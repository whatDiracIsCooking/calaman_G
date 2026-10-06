---
name: test
description: >-
  Run this repo's suites the right way and interpret the result honestly — the
  C++ tier (devtools/cpp-tier.sh, configure + build + ctest), the Python tier
  (pytest), the package tier (devtools/install-check.sh, currently dormant), the
  cross-backend compile check (devtools/cross-backend-check.sh, does the OTHER
  GPU backend still build) and coverage (devtools/coverage.sh) — use the right
  worker count for host vs container, and ALWAYS audit what skipped, or was
  never there, before calling a run green.
  Use when the user asks to run tests, check a change, or verify an edit.
---

# Running tests, and trusting the result

Two rules, and the second is about where this repo's numbers are proved:

1. **A green run is meaningless until you read the skip list.** Tests that need
   a tool the environment lacks skip *silently*, so "all passed" can mean "the
   tests that would have caught this never ran." Always pass `-rs`. On a bare
   host that is the *normal* case here — the whole C++ toolchain lives in the
   CUDA image.
2. **Only a run on a card proves a number.** `src/` and `test/` are live: the
   numerical suites compare each routine's device result against the reference
   LAPACK, and they carry the `gpu` ctest label. A green `cpp-tier.sh` on a box
   with a GPU **does** assert this project's numbers, for the backend it built.
   `ctest -LE gpu` — the `ci-cuda`/`ci-hip` presets, and so all of CI — excludes
   every one of them and proves compile-and-link plus the host-only suites. Say
   which of the two you ran.

## Two suites, and neither covers the other

| | the C++ suite | the Python suite |
|---|---|---|
| Where | `test/` — one suite per `src/` module, plus `test/shared/` | `.claude/hooks/` (the only pytest path) |
| Language | C++23, googletest | Python, pytest |
| Runner | `ctest` via `devtools/cpp-tier.sh` | `pytest` |
| Tests | the library's own units, incl. device kernels | one checker: `.claude/hooks/protect-main.py` — no built binary |
| Needs | the CUDA image (clang-20, CMake 4.2, nvcc) + a card for anything numerical | a venv |
| Gated by | nothing automatic — see below | the pre-push hook, only on a `.py` |

There is **no top-level `tests/` directory** — each test lives next to what it
tests, and `testpaths` in `pyproject.toml` names `.claude/hooks` only — `test/`
is C++ and has no pytest files. Add a directory to `testpaths` in the same
commit that lands its first Python test: an entry pointing at a directory that
does not exist is a pytest **error**, not a skip.

**Which suite a change needs is not negotiable by convenience.** A change under
`src/`, `deps/`, `cmake/` or `CMakeLists.txt` is verified by the C++ tier and by
nothing else; a green `pytest` on it means only that the guard script still
imports. Say which suite you ran.

Two more things that are not suites: `devtools/install-check.sh` (the *installed
package*) and `devtools/cross-backend-check.sh` (does the OTHER backend
compile). Both below.

## The C++ tier

```bash
devtools/cpp-tier.sh                       # configure + build + ctest, default preset
devtools/cpp-tier.sh --preset debug        # switches the ctest preset too
devtools/cpp-tier.sh --preset asan        # hip-asan for the ROCm twin
devtools/cpp-tier.sh --preset compute-sanitizer   # CUDA only, memcheck per suite
devtools/cpp-tier.sh --fresh               # wipe the CMake cache and reconfigure
devtools/cpp-tier.sh --no-test             # configure + build only
devtools/cpp-tier.sh --clean               # rebuild every object, keep the cache
devtools/cpp-tier.sh --tidy                # + clang-tidy over src/*.cpp, ADVISORY
devtools/cpp-tier.sh -j 8                  # parallel compile jobs (BUILD_JOBS)
devtools/cpp-tier.sh -- -R Getrf           # everything after -- goes to ctest
```

Run it **inside the container**, which is where the toolchain is:

```bash
devtools/devcontainer.sh shell -c devtools/cpp-tier.sh
```

On a host with no `cmake`/`ninja` the script refuses with exactly that command
rather than half-running — take the refusal at face value.

Things that will bite:

- **It builds WarpWraps too.** The dependency is a submodule built from source, and
  with `add_subdirectory` consumption its compile-time tier and examples build
  alongside this project (`deps/CMakeLists.txt` says why). So a cold run is
  dominated by the dependency, a warm one is not, and **a compile error may well
  be in `deps/WarpWraps/` rather than in your change** — read the path in the
  diagnostic before assuming it is yours. `sccache` is what makes this
  bearable — it caches the module-interface compiles, which is most of what a
  cold run spends its time on; `doctor.sh` warns when it is absent.
- **A missing submodule is the first thing to check** on a fresh clone:
  `git submodule update --init --recursive`. `doctor.sh` reports it.
- Only `default`, `workstation`, `debug`, `asan`, `compute-sanitizer`, `hip`,
  `hip-asan`, `compile-time`, `coverage`, `ci-cuda` and `ci-hip` have **test**
  presets. With any other configure preset the script builds and then reports
  that there is nothing to ctest — which is not the same as passing.
- `default` and `workstation` are the **same configuration**, differing only in
  `binaryDir` (`build/` vs `build-workstation/`), so a container build and a
  host build can coexist instead of reconfiguring each other.
- `CMAKE_CUDA_ARCHITECTURES=native` queries a live device **at configure time**.
  No GPU, no configure — even for a CPU-only change. `ci-cuda` pins sm_86 and is
  the preset to reach for on a box with no card.
- Every run tees to `.slow-tier-reports/cpp-<stamp>-<rev>-<preset>.log` and
  appends a PASS/FAIL line to `summary.log`. Quote the log path when reporting a
  failure; it has the compiler diagnostics that the summary does not.
- **ctest reports one entry per gtest SUITE**, registered by
  `calaman_add_gtest_suite_tests()` as an `add_test` with a `--gtest_filter`.
  Nothing uses `gtest_discover_tests`, so `-R` selects by suite name and there is
  no test-time discovery step.
- **Deliberately no case counts quoted in this skill.** The suite grows with
  every module PR, so a number written here would be stale within days, and a
  stale count is worse than none. Count from the run in front of you
  (`ctest -N`, and `ctest -N -L gpu` for the labelled share).

### Two things that FAIL rather than skip

Both matter more here than in a pure-wrapper project, because this one computes
numbers:

- **No device.** Nothing is expected to call `GTEST_SKIP` on a missing card, so a
  GPU-less box *fails* the device suites. The way to run without one is the `gpu`
  ctest label — `ctest -LE gpu`, which the `ci-cuda` and `ci-hip` test presets
  do — because an exclusion is named in the output where a skip blends into
  green. Label every suite that touches a device.
- **No CPU reference LAPACK.** `CMakeLists.txt` reports a `WARNING` at configure
  time and does not define `calaman::lapack_reference`. A test that compares a
  factorisation against the reference must be conditioned on that target
  existing, so its absence is a *missing tier* you can see rather than a suite
  that quietly asserts nothing. If a configure log has that warning, no
  numerical comparison ran — say so.

## The Python tier: there is only one

```bash
pytest -n auto -rs          # the whole Python suite, under two seconds
pytest -n auto -rs -k name  # narrower
```

No fast/slow split: the Python side is checker scripts, not numerics, so
`FAST_TEST_ARGS` in `devtools/config.sh` is empty and still prepended to the
default run. If you reintroduce a marker there, add something that runs the other
half in the same change — a marker with no runner is a tier that silently never
executes.

## The cross-backend check, when the change touches a `.cu`

```bash
devtools/cross-backend-check.sh               # compile the OTHER backend, docker included
devtools/cross-backend-check.sh --fresh       # wipe its CMake cache first
devtools/cross-backend-check.sh --device-only # only the device .cu libraries
```

Run it from the **host** — unlike every other tier, this one is not run inside
the dev container. It does its own one-shot `docker run` against the `hip`
image, because the CUDA container it would otherwise run in has no ROCm at all.

**Run it whenever the change touches a `.cu`, or anything under `src/` a `.cu`
includes.** A `.cu` is compiled by nvcc under CUDA and clang under HIP, and nvcc
is the more permissive of the two — so a whole class of error passes the C++ tier
green and only appears when someone builds ROCm.

`--device-only` builds `CROSS_CHECK_DEVICE_TARGETS` from `devtools/config.sh`,
which defaults to the `calaman_device_libraries` umbrella: every
`calaman_add_gpu_device_library()` target joins it, so a new `.cu` library is
covered with no list to update. It catches a kernel-side divergence but not one
in a module unit — the default whole-tier run is the stronger check. If the
variable is overridden empty the flag refuses and exits 2 rather than building
nothing and reporting a pass.

Reading the result:

| Exit | Means |
|---|---|
| 0 | the other backend compiles |
| 1 | it does not — the compiler error is in the output |
| 2 | the check could not run (no `hipconfig`, no docker, no `hip` image, or `--device-only` with nothing to build) |

**Exit 2 is not a pass.** It prints the `docker/build.sh hip` command it needs.
Never report a run that exited 2 as a green cross-backend check.

It proves the other backend **compiles**, not that it runs. `devtools/cpp-tier.sh
--preset hip` on a real ROCm box is the stronger statement.

## The package tier — dormant, and say so

```bash
devtools/install-check.sh              # install to a temp prefix, then consume it
devtools/install-check.sh --no-run     # build the consumer, do not run it (no GPU)
```

**It cannot pass today, and that is expected.** `CALAMAN_INSTALL` defaults OFF
because WarpWraps is consumed with `add_subdirectory`, so this project's export set
would name targets that no package exports; the top-level `CMakeLists.txt` option
comment states the two ways out. Until one is chosen, this tier and CI's
`install-check` job have nothing to prove — report it as *not applicable*, never
as green.

What it will be for: a consumer of a C++23 module package *compiles the installed
module sources*, so a `.cppm` with a `PRIVATE` include directory or define builds
perfectly in-tree, installs without error, and then fails in every consumer.
`cpp-tier.sh` builds in place and never installs, so no amount of green there
says anything about that.

## Coverage

```bash
devtools/coverage.sh                   # configure + build + ctest + llvm-cov, build-coverage/
```

`COVERAGE_IGNORE_REGEX` in `devtools/config.sh` drops `/deps/` — WarpWraps and
GoogleTest — from the summary, so the number is about this project. Most of
`src/` is exercised only by `gpu`-labelled suites, so a coverage run without a
card measures the host-only suites alone — say which it was before quoting a
percentage.

## Host vs container — the worker count differs

- **Host:** `pytest -n auto -rs` from the repo root.
- **Container:** `devtools/devcontainer.sh test` runs the config's `TEST_CMD`
  with `-n $JOBS -rs` inside. `JOBS` defaults to 8 rather than `auto` on purpose
  — several worktree containers each claiming every core is how a box ends up
  thrashing — and every config value honours an environment override, so give a
  concurrent run its own count on the *host* side of the call:

  ```bash
  JOBS=4 devtools/devcontainer.sh test        # -> pytest -n 4 -rs
  devtools/devcontainer.sh test -k some_test  # extra pytest args pass through
  ```

  `JOBS` bounds pytest and nothing else. To bound the *container* — so a
  concurrent run in another worktree cannot steal the cores, and a timing number
  means something — pin it with `CPUSET`/`CPUS` on `up`; see the `devbox` skill.
  **Any LAPACK benchmark without those pins is not a measurement.**

For the **Python** suite the container runs exactly what the host runs. The
container's advantage is the **C++** tier, which cannot run on a bare host at
all. So a green `pytest` on a laptop is the full Python result and still says
nothing about `src/`.

## The markers: `gpu` and `no_sanitizer`

`pyproject.toml` declares two markers, and **nothing currently uses either** —
they are the vocabulary for tests that need them.

- `gpu` — needs a device. These skip on any host without `nvidia-smi`, silently
  unless you passed `-rs`. **This is the marker that would make a laptop run look
  green.** Note the C++ side uses a ctest *label* of the same name, which
  excludes by name instead; prefer that for anything that computes.
- `no_sanitizer` — deselected by `docker compose run --rm compute-sanitizer`,
  because instrumentation turns a merely-slow case into an hours-long one.

## The batch path

`docker/compose.yaml` runs both suites in one shot, gtest then pytest, logging
to `.log/`:

```bash
export HOST_UID=$(id -u) HOST_GID=$(id -g)
docker compose -f docker/compose.yaml run --rm test
docker compose -f docker/compose.yaml run --rm asan
docker compose -f docker/compose.yaml run --rm compute-sanitizer
```

`SKIP_GTEST=1` / `SKIP_PYTEST=1` narrow it to one suite, `TEST_FILTER` is a
regex for `ctest -R` and `PYTEST_ARGS` is appended to pytest. Prefer this when
you want a clean throwaway run; prefer `cpp-tier.sh` inside the devcontainer when
you are iterating. See the `devbox` skill for the difference.

## After the run: audit the skips, then the workers

1. **Read the `-rs` skip reasons.** Separate the expected ones from "that
   toolchain is absent and its coverage did not run" — and say the latter
   explicitly rather than claiming the change is verified. `devtools/doctor.sh`
   reports the whole tool list in one command.

2. **If you interrupted a container run, check for orphaned workers.** Killing
   `devtools/devcontainer.sh test` on the host kills only npx — the pytest master
   and its xdist workers keep running inside at 100% CPU, and nothing tells you;
   the next run just comes out slower and you measure *that*:

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

And say what did *not* run at all. For a C++ run, name the preset and backend,
the entry count, and how many of those were `gpu`-labelled:

- On a card: "`cpp-tier.sh --preset <p>` (<CUDA|HIP>, <device>): N/N ctest
  entries passed, G of them `gpu`-labelled numerical suites checked against the
  reference LAPACK." Say if the *other* backend was not run.
- Without one (`ctest -LE gpu`, or CI): "compile-and-link for <backend> plus
  the M host-only entries; the G `gpu` suites were excluded, so no device
  number was checked."

If the configure log has the missing-LAPACK `WARNING`, the numerical suites
asserted nothing — say that too. A change under `src/` reported as "tests pass"
on a pytest-only or `-LE gpu` run is the failure this skill exists to prevent.
