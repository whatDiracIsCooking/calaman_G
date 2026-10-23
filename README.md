# calaman_G

GPU-accelerated dense linear algebra and solvers in C++23 named modules —
LAPACK-shaped work (factorisations, linear solves, least squares, eigenproblems)
on the device, checked against the netlib reference LAPACK on the CPU.

**One source tree, either vendor.** NVIDIA and AMD support does not live here: it
lives in [WarpWraps](https://github.com/whatDiracIsCooking/WarpWraps), a
submodule at `deps/WarpWraps`, which exposes the CUDA and HIP APIs as importable
modules and maps its backend-neutral `wwr*` names onto whichever one a build
selected. So there is no `src/cuda` and no `src/hip` here — `src/` is written once
and compiles for both. A build targets exactly one backend
(`CALAMAN_GPU_BACKEND=CUDA|HIP`): a HIP build needs no CUDA toolkit, and a CUDA
build needs no ROCm.

> **Status: the library builds and runs; it does not install yet.** `src/` and
> `test/` are live and GPU-exercised — ~130 module libraries, 100+ routine
> directories under `src/lapack/`, and ~130 GoogleTest executables that compare
> device results against the reference LAPACK on a real card. What is still
> dormant is the **install tier**: `CALAMAN_INSTALL` defaults OFF and
> `find_package(calaman)` is not a thing yet (`docs/architecture.md` §2). Consume
> this with `add_subdirectory` for now.

## What is in it

**One module per LAPACK routine, one routine per directory**, and a routine's
call graph is expressed as real module imports rather than partitions of an
umbrella module — `calaman.laqp2` imports `calaman.larfg` and `calaman.larf`,
`calaman.geqp3` imports `calaman.laqp2` and `calaman.laqps`, and so on down.

| | Routines |
|---|---|
| Eigenproblems | general real (`?geev`), symmetric/Hermitian (`?syev`/`?heev`), the Hessenberg→Schur chain (`?gehrd`, `?hseqr`, `?laqr0`…`?laqr5`, `?trevc3`, `?trexc`), tridiagonal (`?steqr`, `?sterf`), balancing (`?gebal`/`?gebak`) |
| Linear solves | positive definite (`?posv`), symmetric and Hermitian indefinite (`?sysv`/`?hesv` + `?sytrs`/`?hetrs`/`?*trs2`), pivoted Cholesky (`?pstrf`/`?pstf2`) |
| Orthogonal factorisations | column-pivoted QR (`?geqp3`, `?laqp2`, `?laqps`), Householder machinery (`?larf`, `?larfg`, `?larft`, `?larfb`, `?larfx`), `?orghr`/`?ormhr` |
| Norms, scaling, utilities | the `?lan*` family across every storage form, `?lascl`/`?rscl`/`?lassq`, RFP↔packed↔full conversions (`?tfttr`, `?tpttf`, `?trttp`, …), `?larnv`/`?laruv` |
| Beyond LAPACK | the matrix exponential (`expm` + Paterson–Stockmeyer + Horner), the `feast` eigensolver, `nnls`, inverse square root, and an iterative/Krylov tier (Lanczos, Davidson, shifted COCG, shift-invert, DIIS, orthogonalisation, Ritz selection) |

Each one has its own suite under `test/`. The numerical suites are labelled
`REQUIRES_GPU`, so `ctest -LE gpu` keeps the handful of host-only suites and
excludes everything that needs a card.

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

**CI cannot run a kernel.** Both backends are built on GitHub-hosted runners and
run `ctest -LE gpu` — compile-and-link plus the host-only suites — so every
numerical claim this project makes rests on `devtools/cpp-tier.sh` on a box with
a GPU. Run it before a PR that touches device behaviour, and say what you ran.

## Layout

| Path | What it is |
|---|---|
| `src/` | the library: one backend-neutral tree, one module per routine |
| `test/` | its suites, one per module, checked against the reference LAPACK |
| `experimental/` | unstable tier, gated behind `CALAMAN_BUILD_EXPERIMENTAL` (default OFF) |
| `deps/` | the WarpWraps submodule and the GoogleTest fetch |
| `cmake/` | the macros every target is declared through |
| `docker/`, `.devcontainer/` | the images, in a four-file diamond |
| `devtools/` | the tiers: `cpp-tier.sh`, `cross-backend-check.sh`, `install-check.sh`, `coverage.sh`, `doctor.sh`, and the container/worktree drivers |
| `docs/` | `architecture.md` (decisions, dated), `CONTRIBUTING.md` (the map), `sanitizers.md` |

## Where to read next

- **`.claude/CLAUDE.md`** — how to work in this repo, and why each piece is shaped
  the way it is. The long answer to almost any question here.
- **`docs/CONTRIBUTING.md`** — setup, the gates, and what to run before a PR.
- **`docs/architecture.md`** — decisions this project made.
- **`deps/WarpWraps/docs/architecture.md`** — the CUDA-vs-HIP vendor facts, which
  belong to the dependency and are not restated here.

## License

MIT — see `LICENSE`.
