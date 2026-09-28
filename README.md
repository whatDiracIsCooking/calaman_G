# calaman_G

GPU-accelerated dense linear algebra and solvers in C++23 named modules —
LAPACK-shaped work (factorisations, linear solves, least squares, eigenproblems)
on the device, checked against the netlib reference LAPACK on the CPU.

**One source tree, either vendor.** NVIDIA and AMD support does not live here: it
lives in [gpumod / Warp Wraps](https://github.com/whatDiracIsCooking/gpumod), a
submodule at `deps/gpumod`, which exposes the CUDA and HIP APIs as importable
modules and maps its backend-neutral `wwr*` names onto whichever one a build
selected. So there is no `src/cuda` and no `src/hip` here — `src/` is written once
and compiles for both. A build targets exactly one backend
(`CALAMAN_GPU_BACKEND=CUDA|HIP`): a HIP build needs no CUDA toolkit, and a CUDA
build needs no ROCm.

> **Status: scaffolding.** The toolchain, containers, dependency wiring and dev
> tooling are in place and `src/` is empty — there is no library to use yet.

## Build

The toolchain (clang-20 with libc++'s module manifest, CMake 4.2, a vendor SDK,
the reference LAPACK) is not expected to be on your host; `docker/` builds it.

```bash
git submodule update --init --recursive
uv sync
devtools/devcontainer.sh rebuild                      # build + start the container
devtools/devcontainer.sh shell -c devtools/cpp-tier.sh # configure, build, ctest
```

For the AMD backend, `devtools/devcontainer.sh --hip shell`. Either way,
`devtools/doctor.sh` is the first thing to run when something behaves oddly.

## Layout

| Path | What it is |
|---|---|
| `src/` | the library: one backend-neutral tree (empty today) |
| `test/` | its suites (empty today) |
| `deps/` | the gpumod submodule and the GoogleTest fetch |
| `cmake/` | the macros every target is declared through |
| `docker/`, `.devcontainer/` | the images, in a four-file diamond |
| `devtools/` | the tiers: `cpp-tier.sh`, `cross-backend-check.sh`, `install-check.sh`, `coverage.sh`, `doctor.sh`, and the container/worktree drivers |
| `docs/` | `architecture.md` (decisions, dated) and `CONTRIBUTING.md` (the map) |

## Where to read next

- **`.claude/CLAUDE.md`** — how to work in this repo, and why each piece is shaped
  the way it is. The long answer to almost any question here.
- **`docs/CONTRIBUTING.md`** — setup, the gates, and what to run before a PR.
- **`docs/architecture.md`** — decisions this project made.
- **`deps/gpumod/docs/architecture.md`** — the CUDA-vs-HIP vendor facts, which
  belong to the dependency and are not restated here.

## License

MIT — see `LICENSE`.
