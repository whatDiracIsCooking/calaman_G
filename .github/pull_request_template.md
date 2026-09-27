<!-- The diff is already in the PR. This body is what a reader gets six months
     from now, so spend it on why rather than on what. -->

## What and why

## What was run

CI gates compile, link, export and every non-device test **on both backends**.
What it cannot run is the seven `gpu`-labelled suites, because no hosted runner
has a card. Tick what you ran locally:

- [ ] `devtools/cpp-tier.sh` — the full tier on a real GPU. **The only thing
      that runs the device-dependent suites**, so required for anything
      touching device behaviour.
- [ ] `devtools/cross-backend-check.sh` — ~7s; does the *other* backend still
      compile. (Superseded by CI's `cpp (hip)` leg, still the fast local answer.)
- [ ] `devtools/install-check.sh` — the one regression nothing else sees: a
      `PRIVATE` include directory or define on a `.cppm`.
- [ ] `pytest -n auto -rs` — the Python tier. Say what skipped; a silent skip
      looks green.
- [ ] Nothing local — this touches no `src/`, no `.cu` and no build files.

## Notes for the reviewer

<!-- Delete what does not apply. -->

- **Touches `docker/`?** The Dockerfiles are now built (not pushed) on PRs that
  change them, so one that no longer builds fails here. The `cpp` legs are
  still resolved against the image `main` already published — to test the new
  image, run the `images` workflow on this branch first (Actions → images → Run
  workflow → pick the branch), then push again.
- **Moved a pinned number, a threshold or a hand-written list?** Say which way
  it moved and why. A loosened check is green by construction — the
  `SuiteListIsComplete` guards and the dispatch TOML tables are the cases here.
- **Changed what a doc claims?** README, CLAUDE.md and `docs/architecture.md`
  each state facts about this tree; update the one that went stale rather than
  leaving two answers.
