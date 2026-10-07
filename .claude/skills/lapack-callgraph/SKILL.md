---
name: lapack-callgraph
description: >-
  Map a reference-LAPACK routine's transitive call graph and which of its
  callees calaman already ships, with devtools/lapack-callgraph.sh. Use before
  porting a routine ("what does DGEQRF need", "what's missing for ZHETRF"),
  when scoping a milestone of module PRs from a driver down, when deciding
  the module-import edges between routines, or when asking whether netlib's
  source matches GitHub's for a routine. Reads Fortran source; needs network
  on a cold cache, no GPU and no build.
---

# LAPACK call graphs

`devtools/lapack-callgraph.sh` walks a reference-LAPACK routine's Fortran
source breadth-first and prints every routine it reaches, then a **gap
report**: which LAPACK (`SRC`) callees already have a `src/lapack/<name>/`
module here, and which do not. It touches no tier — no build, no card — so
say "read the reference source", not "tested", when you report from it.

## Run it

```bash
devtools/lapack-callgraph.sh DGEQP3            # edges, then the calaman gap
devtools/lapack-callgraph.sh --edges ZGEQP3    # edges only
LAPACK_REF=v3.12.0 devtools/lapack-callgraph.sh DGEQP3   # pin the source
devtools/lapack-callgraph.sh --netlib DGEQP3   # + diff against netlib's tarball
devtools/lapack-callgraph.sh --refresh DGEQP3  # drop the cache, re-download
```

The name is case-insensitive and must carry its precision prefix (`DGEQRF`,
not `GEQRF`). Options go **before or after** the name, but `LAPACK_REF` is an
environment variable — `--edges LAPACK_REF=v3.12.0` is read as the routine
name and fails with `cannot fetch`. A cold walk of a driver takes ~15s
(one `curl` per routine); a warm one is instant.

## Pin `LAPACK_REF` to the oracle — the default is `master`

The tests check against the distro LAPACK in the image (Ubuntu 24.04:
**3.12.0**, `docker/Dockerfile.base`), but the script defaults to GitHub
`master`, which has moved on. The graphs differ in ways that change the work:
at `master`, DGEQP3 reaches `DLARF1F` and `DLARFT_LVL2`; at `v3.12.0` it
reaches `DLARF` and neither of those exist. A gap list built at `master` can
ask you to port a routine the oracle does not have — so there is nothing to
test it against.

- **Scoping a port that the test suite will check:** use
  `LAPACK_REF=v3.12.0`. Confirm the oracle's version with
  `dpkg -s liblapack3 | grep Version` if the image may have moved.
- **Looking ahead** (what will a future LAPACK need): `master` is fine — say
  which ref you used.

Each ref has its own cache (`~/.cache/calaman/lapack-src/<ref>`, or
`LAPACK_SRC_CACHE`), so switching refs never mixes sources.

## Reading the output

**Edges** — one `ROUTINE -> CALLEE ...` line per routine reached, root first.
A callee is a `CALL` target or an `EXTERNAL` name, so functions (`ILAENV`,
`DLAMCH`, `ILADLC`) appear too. `X MISSING` means no source was found in
`SRC`, `BLAS/SRC` or `INSTALL` at that ref; `BLAS_*_X` ones are XBLAS and
expected.

**The gap** — `have` / `missing` per `SRC` routine. Read it with these limits:

- **`have` means the directory exists, not the precision.** `DLARFG` and
  `ZLARFG` both map to `larfg/`; check the module's instantiations before
  claiming the `z` variant ships.
- **BLAS is left out on purpose** — it is the vendor's (`wwr.blas`), not a
  routine module. So are `ILAENV`/`IEEECK`/`IPARMQ`, the `ILAPREC`-style enum
  helpers, `XERBLA`, `LSAME` and `?ISNAN`.
- **Some `missing` lines are not ports.** `ILADLC`/`ILADLR` (last non-zero
  column/row) and similar tiny `ILA*` helpers usually fold into the device
  routine that calls them rather than becoming their own module; level-2
  routines (`DGEQR2`, `DORM2R`) may be subsumed by a blocked one. Treat the
  list as candidates, then decide per routine.

**`--netlib`** appends one line per routine whose netlib-tarball source
differs from the GitHub ref: absent on one side, different callees, or
different executable statements (`only IMPLICIT NONE added` is noise from a
tree-wide master change). It names netlib's version, which can differ from
both `LAPACK_REF` and the distro oracle. It downloads a ~8 MB tarball once
per cache.

## Using it to plan work

- **Turn the gap into a DAG.** The edges are the import graph: a missing
  routine is ready once everything it calls (minus BLAS/plumbing) is `have`.
  Leaves first; that ordering is what the **milestone** skill wants as issue
  dependencies, and the inter-routine edges become real module imports
  (`laqp2` imports `larfg` + `larf`), not partitions of one module.
- **Precision families share a module.** Run the `D` routine for the real
  graph and the `Z` one for the complex graph — the complex side can add
  callees (`ZGEQP3` reaches `ZLACGV`) the real side does not have.
- **Check the vendor first.** A gap routine the vendor solver already exposes
  (e.g. `getrf`, `sytrf` via `wwr.solver`) may not need a hand-written module;
  the reverse also happens (there is no vendor `hetrf`).

Exit status is 0 when the walk finished, 1 on a usage error or when the root
itself cannot be fetched (typo, wrong ref, no network). A non-root routine
that cannot be fetched is reported `MISSING` and the walk continues.
