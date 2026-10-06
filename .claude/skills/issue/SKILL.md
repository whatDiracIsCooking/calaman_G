---
name: issue
description: >-
  Resolve one GitHub issue end to end: read it, fork a fresh
  .claude/worktrees/ worktree, implement the fix, build AND run the relevant
  suites on both the NVIDIA (CUDA) and AMD (HIP) cards, then `/pr full` and
  merge. Use when the user invokes `/issue <number|url>` or says "resolve issue
  #N in a new worktree, build+test amd+nvidia, /pr full, you may merge".
  Invoking it is the user's merge grant for that one PR. Requires gh + GH_TOKEN.
---

# `/issue <N | issue URL>` — issue → worktree → both cards → merged PR

This skill is the user's standing kickoff, written down once:

> https://github.com/whatDiracIsCooking/calaman_G/issues/N — resolve in a new
> worktree; build+test AMD + NVIDIA; after the relevant tests pass, `/pr full`;
> you have my permission to merge.

**Invoking `/issue` is that permission** — to merge *this issue's* PR, once its
gates below are green. It does not extend to any other PR, to `--admin`, or to
merging red. If the auto-mode classifier still blocks the merge, stop and hand
back the green PR link; do not work around the block.

One issue per invocation. Several numbers → do them one after another, or ask
whether to fan out (the parallel-wave shape has its own gotchas: shared
`src/CMakeLists.txt` slots, rebase races).

## 1. Read the issue — and decide whether it is ready

```bash
gh issue view <N> --comments
```

Accept a bare number, `#N`, or the full URL; the number is all you need.

Stop and ask, rather than start, when:

- the issue is **closed**, or an open PR already references it
  (`gh pr list --search "<N> in:body" --state all`);
- it says **blocked by / depends on** an issue that is still open;
- the acceptance criteria are genuinely ambiguous — a question the code cannot
  answer. A missing detail with a conventional answer is not ambiguous: pick it
  and say so in the PR.

## 2. Fork a fresh worktree from `origin/main`

Per the **worktree** skill, from the main checkout root. Name it
`issue-<N>-<slug>` (slug: 2–4 words from the title, e.g. `issue-222-trttf`) —
never a bare word like `cleanup…` that reads as a verb.

```bash
git fetch origin main --quiet
devtools/worktree.sh sync                     # keep local main honest too
git worktree add .claude/worktrees/<name> -b <name> origin/main
git branch --unset-upstream <name>            # else it tracks origin/main
git -C .claude/worktrees/<name> submodule update --init --recursive
```

Then `EnterWorktree {"path": ".claude/worktrees/<name>"}`. A worktree does not
inherit `deps/WarpWraps` or a build directory; it configures from scratch into
its own `build/` and `build-hip/` (sccache makes that cheap).

## 3. Implement

Read the code the issue names before writing. The house rules that bite here:

- **Look at a sibling.** A new LAPACK module follows the shape of an existing
  one (`src/lapack/<routine>/` or `src/<module>/`, plus its suite under
  `test/`) and adds itself to the parent `CMakeLists.txt` on both sides.
- Apply **codestyle** (includes, constant names), **docstyle** (header and
  declaration budgets — run its budget check), and **workspace** (if the
  routine takes device scratch) while writing, not afterwards.
- Backend-neutral: `wwr*` names only; no `cu*`/`hip*` outside a switch that
  answers for both.
- The reference LAPACK is the oracle, linked by tests only. Never loosen a
  tolerance in `test/shared/tolerance.cppm` (or locally) to get green — that is
  a "when NOT to merge" case.

## 4. Build and test on BOTH cards

Check both cards are alive first — a dead card is a skipped tier, not a pass:

```bash
nvidia-smi -L        # RTX 3080; NVML "Unknown Error" is intermittent
rocminfo | grep -m1 gfx   # gfx1200
```

Then, per the **test** skill, run the relevant suites on each backend — this
runs the kernels against the reference LAPACK, which CI cannot do:

```bash
devtools/cpp-tier.sh -- -R '<Suite>'                  # NVIDIA, default preset
devtools/cpp-tier.sh --preset hip -- -R '<Suite>'     # AMD
```

"Relevant" means the suites for every module the diff touches **and** every
module that imports one of them (`grep -rl 'import calaman.<mod>' src test`).
A change to a shared module (`calaman.common`, `test/shared/`, `cmake/`) or the
WarpWraps pin makes the whole tier relevant: drop the `-R`.

Both legs must build clean and pass, with the skip list read. If one card is
down, say so plainly and do **not** merge on one backend's word — hand back the
PR. Compile-only (`--no-test`) is not "tested".

Before committing, pre-flight lint that `cpp-tier.sh` does not run:

```bash
uv run cmake-lint <every touched */CMakeLists.txt>    # 80-col, CI's exact tool
```

## 5. Ship: `/pr full`

Invoke the **pr** skill with `full`, with these specifics:

- PR body from `.github/pull_request_template.md` via `--body-file` (written to
  the scratchpad — a heredoc trips the redirect false-positive). Put
  **`Closes #<N>`** under *What and why*, and under *What was run* name both
  presets, both devices, and the suites with their pass counts.
- Commit and push with `git -C <abs-worktree-path> …` if protect-main
  misfires; never `--no-verify`, never `CLAUDE_ALLOW_MAIN_EDITS`.
- **Wait for the whole CI run, `ci-ok` included**, then `gh pr merge <n>
  --squash`. Not `--auto`: the ruleset does not require `ci-ok`, so auto-merge
  lands before the `cpp (cuda)`/`cpp (hip)` legs finish.
- If `main` moved and the PR conflicts, rebase, re-run the step-4 suites that
  the conflict touched, push, and wait again.
- Teardown: `ExitWorktree {"action": "keep"}`, then — because the worktree
  holds a submodule, which `git worktree remove` refuses — `rm -rf
  .claude/worktrees/<name> && git worktree prune && git branch -D <name>`.
  Confirm the issue closed (`gh issue view <N> --json state`).

## 6. Report

On a clean run, one or two lines: "Merged PR #M (closes #N); CUDA + HIP
`<Suite>` green." Narrate only what deviated — a blocked merge, a dead card, a
rebase, a decision you made on an under-specified issue.
