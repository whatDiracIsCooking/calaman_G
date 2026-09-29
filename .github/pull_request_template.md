<!-- The diff is already in the PR. This body is what a reader gets six months
     from now, so spend it on why rather than on what. -->

## What and why

## What was run

CI gates compile and link **on both backends**, on runners with no GPU. What it
cannot do is run a kernel — which is to say it cannot check a single number this
library computes. Tick what you ran locally:

- [ ] `devtools/cpp-tier.sh` — the full tier on a real GPU. **The only thing that
      runs the `gpu`-labelled suites**, so required for anything touching device
      behaviour or numerical results. Say which preset, and which backend.
- [ ] `devtools/cross-backend-check.sh` — does the *other* backend still compile.
      (Overlaps CI's `cpp (hip)` leg; still the fast local answer.)
- [ ] `pytest -n auto -rs` — the Python tier. Say what skipped; a silent skip
      looks green.
- [ ] Nothing local — this touches no `src/`, no `.cu`, no build files and not
      the submodule.

<!-- devtools/install-check.sh is DORMANT: CALAMAN_INSTALL is OFF and there is no
     example/consumer yet (docs/architecture.md section 2). It exits 2, "could
     not run". Do not tick it as a pass; restore this line when the tier is
     live. -->

## Notes for the reviewer

<!-- Delete what does not apply. -->

- **Bumped `deps/WarpWraps`?** Name the commit range and say what in its diff
  reaches this project. A bump is a code change: it can move a `wwr*` spelling,
  and CI proves only that both backends still compile at the new pin.
- **Touches `docker/`?** The Dockerfiles are built (not pushed) on PRs that
  change them, so one that no longer builds fails here. The `cpp` legs are still
  resolved against the image `main` already published — to test the new image,
  run the `images` workflow on this branch first (Actions → images → Run workflow
  → pick the branch), then push again.
- **Touches the CPU reference LAPACK, or a test that uses it?** Say whether the
  configure log carried `CPU reference LAPACK not found`. If it did, no
  comparison against a reference result was built, and a green run means less
  than it looks.
- **Moved a pinned number, a threshold or a hand-written list?** Say which way it
  moved and why. A loosened check is green by construction, and a numerical
  tolerance is the easiest thing in this project to loosen by accident.
- **Changed what a doc claims?** `README.md`, `.claude/CLAUDE.md` and
  `docs/architecture.md` each state facts about this tree; update the one that
  went stale rather than leaving two answers. Vendor facts belong to WarpWraps, not
  to `docs/architecture.md` here.
