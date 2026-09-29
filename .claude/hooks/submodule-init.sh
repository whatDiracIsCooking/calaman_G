#!/usr/bin/env bash
# SessionStart + CwdChanged hook: make sure the WarpWraps submodule
# (deps/WarpWraps) is checked out to its PINNED commit in whatever checkout this
# session is in. A fresh clone or a lightweight .claude/worktrees/ worktree
# starts with an empty submodule directory, and nothing under src/ compiles
# without it -- so populate it on session start and whenever the session enters
# a worktree (CwdChanged).
#
# INIT-ONLY, DELIBERATELY. `--init --recursive` restores the committed pin and
# never touches the gitlink, so it cannot dirty the tree. It does NOT advance to
# WarpWraps main HEAD -- that stays a deliberate `git submodule update --remote`
# bump plus a commit (see deps/CMakeLists.txt and the `branch = main` line in
# .gitmodules that tells --remote which branch to follow).
#
# Best-effort and silent: this must never block a session, so a failure (no
# network on a first clone, say) is swallowed and nothing is written to stdout,
# which SessionStart would otherwise append to the session context.
set -uo pipefail

# The checkout the session is IN: cwd's git root -- which on CwdChanged is the
# worktree we just entered, so its submodule gets populated rather than the main
# checkout's. Fall back to CLAUDE_PROJECT_DIR when cwd is not inside a repo.
root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -z "$root" ] && root="${CLAUDE_PROJECT_DIR:-}"
[ -z "$root" ] && exit 0

git -C "$root" submodule update --init --recursive >/dev/null 2>&1 || true
