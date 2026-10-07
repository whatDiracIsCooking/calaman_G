---
name: milestone
description: >-
  Turn a plan — a roadmap/design .md, or the current conversation — into a
  GitHub milestone plus a DAG of issues, each scoped to a single PR and
  cross-linked by dependency so independent issues can be worked in parallel.
  Use when the user asks to "make a milestone", "break this into issues",
  "plan out the work as issues", or wants a design decomposed into trackable,
  parallelizable GitHub work. Requires gh + GH_TOKEN (present in this repo's
  containers). Creating issues is outward-facing — draft and confirm first.
---

# Plan → milestone → DAG of PR-sized issues

Take a plan and turn it into one GitHub milestone whose issues each map to a
single PR, wired into a dependency DAG so the independent ones can run at once.
The hard part is not the `gh` calls — it is the decomposition. Do that well,
show it, then create.

## 1. Get the plan

The input is one of:

- **A file** — `/milestone path/to/plan.md` (a design doc, proposal, or
  roadmap). Read it in full.
- **The conversation** — `/milestone` with no path. Reconstruct the plan from
  what has been discussed and decided in this session. If the discussion is
  thin or ambiguous, say so and ask for the file rather than inventing scope.

Extract two things: the **overall goal** (becomes the milestone) and the
**discrete pieces of work** (become the issues). If the plan is one
undifferentiated wall, your job in step 2 is to cut it up; if it already has
sections/steps, use them as the starting seams but re-check their size.

## 2. Decompose into PR-sized issues

Each issue must be addressable by **one coherent, reviewable PR**. Calibrate:

- **Too big** if you cannot state the PR's diff in a sentence or two, if it
  touches unrelated subsystems, or if a reviewer would have to context-switch
  mid-review. Split it.
- **Too small** if it cannot stand as its own commit with its own test/check —
  a one-line tweak that only makes sense bundled with its neighbour. Merge it
  up.
- **Just right**: one bounded change, an independently statable acceptance
  check, a named set of files/modules it will touch. That usually means one
  module and its test, landing via the `pr` skill as a single squash-merge.

Write each issue with a consistent body so a future implementer (or subagent)
needs nothing else:

```
## Scope
<one-PR change, stated as the diff it will produce>

## Depends on
- #<n>   (or: "None — ready to start")

## Acceptance
- <the check that proves it done: a test name, a benchmark, a build>

## Files / area
<paths or package the PR will touch>
```

## 3. Wire the DAG — maximise what can run in parallel

Add an edge **A → B** only when B genuinely cannot start until A lands — B
imports A's new symbol, builds on a file A creates, or asserts against an
interface A defines. Do **not** add edges for mere preference of ordering; a
spurious edge serialises work that could have run at once.

**Verify the edges against an authoritative source — do not trust a graph
rebuilt from memory.** When the dependencies mirror an existing structure (a
call graph, an import graph, an API), check each edge against that source, not
your recollection or the plan's own prose: a DAG that reads as plausible is
exactly the thing that is wrong in ways only the source reveals. In this repo
the issue edges *are* the LAPACK call graph, checkable against the
Reference-LAPACK Fortran (`grep CALL` in `SRC/<routine>.f`, or the raw file on
GitHub) — doing so on a `geev` decomposition caught three wrong edges and a
missing dependency that a confident from-memory draft had gotten wrong.

Then compute **waves** (topological layers): wave 0 = every issue with no
unmet dependency, wave 1 = issues whose deps are all in wave 0, and so on.
Wide-and-shallow beats deep-and-narrow — if everything chains off one root,
look for a smaller shared prerequisite to split out so the rest fan out. The
wave map is what you show the user and what tells them what to start today.

## 4. Draft, then get approval — do not create yet

Creating issues is public and awkward to undo. Present the whole plan first:

- the **milestone** title + one-line description;
- a **wave table** — for each issue: proposed title, its wave, its `depends on`
  set, and a one-line scope;
- call out the **wave-0 set** explicitly: "these N can be started in parallel
  immediately."

Get a clear go-ahead (or edits) before any `gh` write. If the user tweaks the
cut, revise the table and re-confirm.

## 5. Create the milestone

No `gh milestone` subcommand exists — use the API. Capture the number.

```bash
gh api repos/{owner}/{repo}/milestones \
  -f title="<milestone title>" \
  -f description="<goal + the wave map, in markdown>" \
  --jq '.number'
```

Put the wave map in the description (markdown renders on the milestone page) so
the DAG is visible without opening every issue.

## 6. Create the issues in topological order

Create dependencies **before** dependents. Then when you write a dependent's
body, its `Depends on #N` already has a real number, and GitHub auto-creates
the back-reference on the dependency's timeline — no second pass needed for the
depends-on direction.

```bash
gh issue create \
  --title "<title>" \
  --milestone "<milestone title>" \
  --body "$(cat <<'EOF'
## Scope
...
## Depends on
- #12
## Acceptance
...
## Files / area
...
EOF
)"
```

Keep a title→number map as you go so later issues can reference earlier ones.
Capture each issue's **REST database id** too, not only its number — the
`blocked_by` API in step 6a needs the id, not the `#number`:

```bash
url=$(gh issue create --title "…" --milestone "$MS_TITLE" --body "…")
num="${url##*/}"                                         # #number (for body refs)
id=$(gh api repos/{owner}/{repo}/issues/"$num" --jq '.id')   # database id (for the API)
```

Notes:

- `--milestone` takes the milestone **title**, which must match what step 5
  created. It is `-m, --milestone name` — a bare number is not accepted.
- `--parent <n>` (gh **≥2.94**) makes a true **sub-issue** — a parent/child
  *tree*, one parent per child, so it **cannot encode a DAG** where a node has
  several blockers. Use it only for genuine containment (e.g. nesting every
  issue under one tracking epic), never for the dependency edges. Check
  `gh --version`; distro packages lag (Ubuntu 24.04 ships 2.45), and without it
  the flag is simply absent.
- The body's `Depends on #N` line is the **portable source of truth** and goes
  in every time — it survives an old gh and reads fine without the API.
- Labels are optional and must exist first (`gh label create`); a `wave-N`
  label is a cheap parallelism signal if the user wants one, but the body's
  `Depends on` is what actually encodes the graph.

## 6a. Wire the DAG natively (blocked_by)

The faithful native encoding of the dependency DAG is GitHub's many-to-many
**issue dependencies**, not sub-issues: each edge is a "blocked by" link, and
the UI then greys out blocked issues and shows Blocking/Blocked-by on each.
There is no gh flag — it is the REST issue-dependencies API. Offer it on top of
the body lines (the body line stays regardless); add it once the user is in.

Do it as a **second pass, after all issues exist**, because each edge needs the
database id of the *blocking* issue. First probe the endpoint read-only — a live
repo returns `[]`, a repo without the feature errors — so you fail fast rather
than half-wire the graph:

```bash
gh api repos/{owner}/{repo}/issues/<any-num>/dependencies/blocked_by   # -> [] if live
```

Then, for each edge "child **blocked_by** blocker", POST the blocker's **id**
(use `-F` so it is sent as an integer, not a string):

```bash
gh api --method POST \
  repos/{owner}/{repo}/issues/<child-num>/dependencies/blocked_by \
  -F issue_id=<blocker-id> --silent
```

GitHub records the reciprocal "blocking" side automatically — POST only the
`blocked_by` direction. Because the whole thing is ~1 create + 1 id-fetch per
issue and 1 POST per edge, drive it from a single script that holds the
`name→{num,id}` map in memory (separate `gh` calls do not share shell state),
and log one line per edge so a failed POST is visible rather than silent.

## 7. Report

Finish with the milestone URL and a short wave summary: which issues (by number
now) are in each wave, and the explicit **"ready now"** list — the wave-0
issues with no blocker. That is the answer to "what can we work on
simultaneously." To execute the milestone, point the user at
`/milestone-run <milestone URL>`. If step 6a ran, confirm the edge count wired with no failures.

## Undo, if it came out wrong

Issue creation is reversible only by hand, so this is the escape hatch:

```bash
gh issue delete <number> --yes                                   # per issue
gh api -X DELETE repos/{owner}/{repo}/milestones/<number>        # the milestone
```

Deleting the milestone does **not** delete its issues — they just detach.
Remove the issues first if the whole plan is being scrapped.
