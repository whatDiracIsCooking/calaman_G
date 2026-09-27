---
name: pr
description: >-
  Ship the current work: commit the relevant changes, push the branch, open a
  GitHub PR with gh, and merge it. Use when the user asks to "PR this", "ship
  it", "commit/push/merge", or otherwise wants the working-tree changes landed
  on main end to end. Requires gh + GH_TOKEN.
---

# Commit → push → PR → merge

Drive the current changes all the way to `main` in one pass. Each step has a
gotcha; follow them in order.

## 1. Commit the relevant changes

- **Never commit on `main`.** If the session is on `main`, create a branch
  first (`git switch -c <topic>`). If the session is in a `.claude/worktrees/`
  worktree, the branch already exists — use it.
- Stage **relevant** changes, not everything: read `git status` and `git diff`,
  stage what belongs to this task, and leave unrelated untracked files behind.
  No blanket `git add -A` without looking.
- The pre-commit hook lints every touched file — a finding blocks the commit;
  fix it, don't `--no-verify`.
- If `pyproject.toml` changed, run `uv lock` and commit the lock with it. The
  lock is the only place exact versions live, and the image installs it with
  `--frozen`, so a stale lock breaks the container rather than the host.

## 2. Check the repo's own invariants

Before pushing, look at what the diff touches and whether this repo guards it:

- a **ratchet** (a test pinning a measured number or a floor) may only move in
  the improving direction, in its own commit, saying what moved. A loosened
  threshold produces a green run — that is exactly why it needs a human.
- the **reference chain** (whatever independently checks correctness here) is
  not something to adjust to make a test pass.

A repo with neither has nothing to do in this step.

## 3. Push

```bash
git push -u origin <branch>
```

- A push touching any `.py` auto-runs the fast test gate
  (`devtools/prepush-tests.sh`). A successful push therefore *implies* that
  gate passed. Do not bypass with `--no-verify`.
- If the worktree's git dir has gone missing (`fatal: not a git repository:
  …/worktrees/<name>`), push from the main checkout instead:
  `git -C <repo-root>/main push origin <branch>`.

## 4. Create the PR

```bash
gh pr create --title "<concise title>" --body-file <path>
```

**Fill in `.github/pull_request_template.md`, do not bypass it.** `gh pr create`
applies the repository template only when no body is supplied, so `--body "…"`
silently discards it — and its checklist is the only record of which tiers
actually ran, the device suites among them, which nothing on a runner can
verify. Copy the template to a scratch file, fill in *What and why*, tick only
the commands you really ran, delete the reviewer notes that do not apply, and
pass it with `--body-file`.

The harness's PR-body footer rules apply.

## 5. Merge

```bash
gh pr merge <number> --squash
git push origin --delete <branch>     # remote branch cleanup
```

- **Wait for CI.** `.github/workflows/ci.yml` runs lint, the Python tier, the
  whole C++ tree for BOTH backends and the install check on every PR, so it
  takes tens of minutes rather than seconds; the local pre-push gate is a small
  subset of it. A PR touching `docker/` also builds the images. Check with
  `gh pr checks <number> --watch` before merging, and do not merge red.
- **Never pass `--admin`.** It merges past required status checks, and it is the
  one move in this whole flow that can defeat the branch ruleset. Red checks
  mean fix them or hand the PR back — never override. A ruleset with an empty
  bypass list makes the server refuse it anyway, but do not lean on that: the
  bypass list is a setting someone can widen, and this rule is the intent.
- **Never pass `--delete-branch`**: it attempts a local `git checkout main`,
  which fails when `main` is held by another worktree. Delete the remote branch
  with `git push origin --delete` instead, as above.
- If GitHub reports the PR as not yet mergeable, wait a moment and re-check
  with `gh pr view <number> --json mergeable,mergeStateStatus` before retrying
  — do not force.

## 6. After the merge

`origin/main` has moved but local `main` has not — and `worktree.sh add` forks
from local HEAD, so a stale `main` silently seeds stale branches:

```bash
devtools/worktree.sh sync
```

It finds whichever checkout has `main` on it and fast-forwards it there, so it
runs from any worktree; it refuses (`--ff-only`) rather than merging if that
`main` has diverged.

If the work happened in a `.claude/worktrees/` worktree, whether to tear it
down depends on how this skill was invoked:

- **Bare `/pr`** — leave the worktree standing. The work is merged, but the
  checkout stays for follow-up commits; do not remove it unless asked.
- **`/pr full`** — reclaim it now, per the `worktree` skill's "`/pr full` —
  ship, then tear the worktree down" section: exit with
  `ExitWorktree {"action": "keep"}` if the session is inside it, then
  `git worktree remove` + `git branch -d <name>` (`-D` after a squash merge).

Otherwise — a plain topic branch on an ordinary checkout — the local branch can
go with `git branch -d <branch>` (or `-D` after a squash merge, since the SHAs
differ; the merge already confirmed the patches landed).

## When NOT to merge

Stop after step 4 and hand the PR to the user instead of merging when:

- CI is red, or the push gate was bypassed with `--no-verify`;
- the diff moves a ratchet floor or touches the correctness references in ways
  only a human should sign off on;
- the user asked for a PR but not a merge.
