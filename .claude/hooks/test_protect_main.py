"""Tests for protect-main.py, the PreToolUse guard beside it.

Same shape of argument as test/shared/test_dispatch.py: a guard's failure mode
is ALLOWING what it should deny, which is silent and looks exactly like working
correctly. Nothing else in the repo exercises it -- it only runs inside a live
Claude Code session, where a hole in it is invisible until someone notices the
primary checkout has been edited.

That is not hypothetical. The guard matched only Write|Edit|NotebookEdit, so a
session editing through `sed -i`, a `>` redirect or a `python3 - <<EOF` heredoc
walked straight past it and rewrote the main checkout -- which is how this file
came to exist. Extending it to Bash meant writing a heuristic over arbitrary
shell, and the first draft got three cases wrong: it waved heredocs through
(the writes land in a different newline-split segment from the `python3` that
introduces them), and it denied both `sed -i /tmp/x` and `git diff > /dev/null`
(a cwd fallback applied even when the paths HAD been read). Each is a line
below.

The payloads are the real PreToolUse contract -- tool_name, cwd, tool_input --
fed to the script as a subprocess, so this tests the hook exactly as the
harness invokes it. Empty stdout means allow; a JSON decision means deny.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parent / "protect-main.py"

# The guard now logs every decision. Point that at a throwaway dir for the whole
# suite so the ~60 invocations below do not spam the real .logs; the tests that
# assert ON the log override this again with their own tmp_path.
_SUITE_LOG_DIR = Path(tempfile.mkdtemp(prefix="protect-main-log-"))


def _primary_worktree():
    """The checkout the guard protects: parent of the shared git common dir."""
    common = subprocess.run(
        ["git", "-C", str(HOOK.parent), "rev-parse",
         "--path-format=absolute", "--git-common-dir"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return Path(common).resolve().parent


MAIN = _primary_worktree()

# The nested carve-out. Addressed as a path rather than a cwd so the case does
# not depend on any particular worktree existing when the suite runs.
NESTED = ".claude/worktrees/probe"

HEREDOC = (
    'python3 - <<PY\n'
    'from pathlib import Path\n'
    'Path("pyproject.toml").write_text("x")\n'
    'PY'
)
NESTED_HEREDOC = (
    'python3 - <<PY\n'
    'from pathlib import Path\n'
    f'Path("{NESTED}/x.py").write_text("1")\n'
    'PY'
)

# (deny?, label, command) -- all run with cwd == the protected checkout.
BASH_CASES = [
    # The bypasses that actually happened.
    (True, "heredoc writing a repo file", HEREDOC),
    (True, "sed -i on a tracked file", "sed -i s/a/b/ devtools/config.sh"),
    (True, "git rm", "git rm -q tests/test_smoke.py"),
    (True, "git mv", "git mv a.py b.py"),
    (True, "redirect into the tree", "echo hi > CLAUDE.md"),
    (True, "cp over a tracked file", "cp /tmp/x README.md"),
    (True, "tee into the tree", "echo x | tee README.md"),
    (True, "rm with a glob", "rm -f *.log"),
    (True, "append into the tree", "echo x >> CLAUDE.md"),
    (True, "mutation naming no path at all", "rm -rf"),

    # Reads, which must stay out of the way.
    (False, "git status", "git status --short"),
    (False, "grep for the word rm", 'grep -rn "rm -rf" devtools/'),
    (False, "sed -n is a read", "sed -n 1,5p CLAUDE.md"),
    (False, "python3 -c with no write", 'python3 -c "print(1)"'),
    (False, "pytest with a 2>&1 dup", "pytest -q 2>&1 | tail -3"),
    (False, "cat", "cat -n devtools/config.sh"),
    (False, "ls", "ls -la src/"),

    # Mutations aimed elsewhere. Both regressed in the first draft, because a
    # cwd fallback fired even though the command's paths had been read fine.
    (False, "sed -i targeting /tmp", "sed -i s/a/b/ /tmp/scratch.txt"),
    (False, "redirect to /dev/null", "git diff > /dev/null"),
    (False, "write under a sibling repo", "cp x.py /home/other/repo/y.py"),

    # Archive tools write only in some modes. `-t` LISTS; it was denied as if
    # it unpacked.
    (False, "tar listing an archive", "tar -tzf archive.tgz"),
    (True, "tar extracting into the tree", "tar -xzf archive.tgz"),
    (False, "unzip listing", "unzip -l archive.zip"),

    # `\.write(` used to match sys.stdout.write, making any script that prints
    # that way look like a file write.
    (False, "script writing to stdout",
     'python3 -c "import sys; sys.stdout.write(\'x\')"'),
    (True, "script writing a repo file", 'python3 -c "open(\'CLAUDE.md\', \'w\')"'),

    # Git-ignored paths are generated and disposable, not the checkout's
    # content. `/build*/` is a DIRECTORY pattern, so these only match once the
    # trailing-slash form is tried too.
    (False, "mkdir of an ignored build dir", "mkdir -p build"),
    (False, "wiping an ignored build dir", "rm -rf build-hip/"),
    (False, "writing into an ignored build dir", "echo x > build/CMakeCache.txt"),
    (False, "writing an ignored report", "echo x > .slow-tier-reports/run.log"),
    (True, "an untracked-but-not-ignored file is still content",
     "echo x > build.log"),

    # The nested-worktree carve-out: this is where work is SUPPOSED to happen.
    (False, "sed -i in a nested worktree", f"sed -i s/a/b/ {NESTED}/config.sh"),
    (False, "heredoc in a nested worktree", NESTED_HEREDOC),
    (False, "redirect into a nested worktree", f"echo hi > {NESTED}/out.txt"),

    # Comments and quoted strings are not shell redirects. An arrow in prose
    # ('squash -> SHAs') matched the old redirect regex as a '>' into a file,
    # which denied the `/pr full` worktree teardown that carried the note, and a
    # '>' inside a quoted argument denied read-only commands that merely quoted
    # one. The teardown verbs themselves rewrite no tree, so they are allowed.
    (False, "arrow in a trailing comment is not a redirect",
     "git worktree remove .claude/worktrees/x\n"
     "# squash merge -> SHAs differ, so -d refuses; -D is safe\n"
     "git branch -D x"),
    (False, "git worktree remove is not a tree rewrite",
     "git worktree remove .claude/worktrees/x"),
    (False, "git branch -D is a ref op, not a tree rewrite", "git branch -D topic"),
    (False, "a gt inside a quoted argument is not a redirect", 'echo "a > b"'),
    (False, "an arrow inside a quoted grep pattern is not a redirect",
     'git log --grep "fix -> x"'),
    (False, "a hash inside quotes is not a comment", 'echo "a # b"'),
    # ...but a real mutation on the same line still lands. These pin that
    # stripping is quote-aware (it did not eat the redirect after a quoted '#')
    # and comment-aware (the redirect BEFORE a trailing comment survives).
    (True, "a redirect before a trailing comment still denies",
     "echo x > CLAUDE.md   # writes the file"),
    (True, "a redirect after a quoted hash still denies",
     'echo "a # b" > CLAUDE.md'),
    (True, "a redirect to a quoted target still denies", 'echo x > "CLAUDE.md"'),
]

# (deny?, label, file_path) -- the original Write/Edit contract.
PATH_CASES = [
    (True, "Write to the protected checkout", str(MAIN / "CLAUDE.md")),
    (True, "Write deep in the protected checkout", str(MAIN / "src/blas.cppm")),
    (False, "Write into a nested worktree", str(MAIN / NESTED / "CLAUDE.md")),
    (False, "Write to /tmp", "/tmp/x.txt"),
    (False, "Write into an ignored build dir", str(MAIN / "build/CMakeCache.txt")),
]


def _decision(tool, cwd, tool_input, log_dir=None):
    """True == denied. The hook prints a JSON decision to deny, nothing to allow."""
    payload = json.dumps(
        {"tool_name": tool, "cwd": str(cwd), "tool_input": tool_input}
    )
    env = {**os.environ,
           "PROTECT_MAIN_LOG_DIR": str(log_dir or _SUITE_LOG_DIR)}
    r = subprocess.run([sys.executable, str(HOOK)], input=payload,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"the guard must never fail closed: {r.stderr}"
    if not r.stdout.strip():
        return False
    decision = json.loads(r.stdout)["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    return True


@pytest.mark.parametrize(
    ("deny", "command"),
    [pytest.param(d, c, id=label) for d, label, c in BASH_CASES],
)
def test_bash_commands(deny, command):
    assert _decision("Bash", MAIN, {"command": command}) is deny


@pytest.mark.parametrize(
    ("deny", "file_path"),
    [pytest.param(d, p, id=label) for d, label, p in PATH_CASES],
)
def test_declared_paths(deny, file_path):
    assert _decision("Write", MAIN, {"file_path": file_path}) is deny


# ---- git operations, judged by what they do rather than what they name ------

GIT_CASES = [
    # Destructive to a tree other sessions share, and naming no path to be
    # caught by. `git clean -fd` was already denied (no readable target, so the
    # cwd fallback fires); the reset was not.
    (True, "reset --hard", "git reset --hard HEAD~1"),
    (True, "reset --merge", "git reset --merge"),
    (True, "clean -fd", "git clean -fd"),

    # Reversible or inert -- guarding these is noise, not safety.
    (False, "plain reset unstages", "git reset HEAD~1"),
    (False, "reset --soft keeps the tree", "git reset --soft HEAD~1"),
    (False, "add is inert", "git add -A"),
    (False, "status", "git status --short"),
    (False, "log", "git log --oneline -1"),
    (False, "branching is the fix, not the problem", "git switch -c topic"),

    # -C moves where git operates; another tree is not our business.
    (False, "reset --hard elsewhere via -C", "git -C /tmp/other reset --hard"),

    # Deliberately NOT guarded: a local hook cannot bind other machines or
    # clients, so blocking this would be false confidence. Branch protection is
    # the real control. Asserted so the omission stays a decision.
    (False, "push is left to branch protection", "git push origin main"),
]


@pytest.mark.parametrize(
    ("deny", "command"),
    [pytest.param(d, c, id=label) for d, label, c in GIT_CASES],
)
def test_git_operations_in_the_primary_checkout(deny, command):
    assert _decision("Bash", MAIN, {"command": command}) is deny


@pytest.mark.parametrize(
    "command", ["git reset --hard HEAD~1", "git clean -fd", "git commit -m x"]
)
def test_git_operations_are_fine_inside_a_worktree(command, tmp_path):
    """A worktree is where this work is supposed to happen.

    Addressed with `-C` at a path under .claude/worktrees/ so the case does not
    depend on a particular worktree existing.
    """
    nested = f"{MAIN}/{NESTED}"
    cmd = f"git -C {nested} {command[4:]}"
    assert _decision("Bash", MAIN, {"command": cmd}) is False


def test_commit_is_blocked_only_while_the_primary_checkout_is_on_main():
    """The repo ships through PRs, and "never commit on main" was only advice.

    Branch-aware rather than hardcoded: a topic branch checked out in the
    primary checkout is ordinary and must stay committable, so this asserts
    against whatever `main` is actually on right now.
    """
    branch = subprocess.run(
        ["git", "-C", str(MAIN), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    denied = _decision("Bash", MAIN, {"command": "git commit -m x"})
    assert denied is (branch == "main"), (
        f"primary checkout is on {branch!r}; commit denied={denied}"
    )


def test_escape_hatch_allows_everything(monkeypatch):
    """CLAUDE_ALLOW_MAIN_EDITS is the documented way to edit main deliberately.

    Checked because a guard with no working escape hatch is one people delete
    rather than configure.
    """
    monkeypatch.setenv("CLAUDE_ALLOW_MAIN_EDITS", "1")
    assert _decision("Bash", MAIN, {"command": "rm -rf src/"}) is False
    assert _decision("Bash", MAIN, {"command": "git reset --hard HEAD~1"}) is False
    assert _decision("Bash", MAIN, {"command": "git commit -m x"}) is False
    assert _decision("Write", MAIN, {"file_path": str(MAIN / "CLAUDE.md")}) is False


# ---- the log: the guard was invisible, so a decision must leave a record ------
#
# A guard you cannot confirm ran is indistinguishable from no guard -- that is
# why an edit reaching main could not be diagnosed. These pin that every decision
# is recorded, and that a bypass is recorded rather than silent.


def _log_lines(log_dir):
    p = log_dir / "protect-main.log"
    return p.read_text().splitlines() if p.exists() else []


def _fields(line):
    ts, decision, session, tool, cwd, *rest = line.split("\t")
    return decision, session, tool, cwd


def test_a_denial_is_recorded(tmp_path):
    assert _decision("Write", MAIN, {"file_path": str(MAIN / "CLAUDE.md")},
                     log_dir=tmp_path) is True
    lines = _log_lines(tmp_path)
    assert len(lines) == 1
    decision, _session, tool, _cwd = _fields(lines[0])
    assert (decision, tool) == ("deny", "Write")


def test_an_allow_is_recorded(tmp_path):
    """So an empty log means "never ran", not "ran and allowed" -- the two the
    incident could not be told apart without this."""
    assert _decision("Bash", MAIN, {"command": "git status --short"},
                     log_dir=tmp_path) is False
    lines = _log_lines(tmp_path)
    assert lines and _fields(lines[-1])[0] == "allow"


def test_a_bypass_is_recorded_and_still_allowed(tmp_path, monkeypatch):
    """The escape hatch allows the edit but must not do so silently: a leaked
    CLAUDE_ALLOW_MAIN_EDITS is precisely what the log exists to surface."""
    monkeypatch.setenv("CLAUDE_ALLOW_MAIN_EDITS", "1")
    assert _decision("Write", MAIN, {"file_path": str(MAIN / "CLAUDE.md")},
                     log_dir=tmp_path) is False
    assert [_fields(line)[0] for line in _log_lines(tmp_path)] == ["bypass"]


def test_the_session_id_is_recorded(tmp_path):
    """The forensic key: the log names WHICH concurrent session edited main."""
    payload = json.dumps({
        "tool_name": "Write", "cwd": str(MAIN), "session_id": "sess-abc123",
        "tool_input": {"file_path": str(MAIN / "CLAUDE.md")},
    })
    env = {**os.environ, "PROTECT_MAIN_LOG_DIR": str(tmp_path)}
    subprocess.run([sys.executable, str(HOOK)], input=payload,
                   capture_output=True, text=True, env=env)
    assert _fields(_log_lines(tmp_path)[0])[1] == "sess-abc123"


def test_relative_file_path_is_resolved_against_the_session_cwd(tmp_path):
    """A relative file_path is rooted at the SESSION's cwd, not this process's.

    os.path.abspath would resolve it against wherever the hook happens to run, so
    a relative path into main would miss. Run with the process cwd set to a
    non-repo tmp dir while the payload's cwd is main: only resolving against the
    payload cwd catches it.
    """
    payload = json.dumps({
        "tool_name": "Write", "cwd": str(MAIN),
        "tool_input": {"file_path": "CLAUDE.md"},
    })
    env = {**os.environ, "PROTECT_MAIN_LOG_DIR": str(tmp_path)}
    r = subprocess.run([sys.executable, str(HOOK)], input=payload,
                       capture_output=True, text=True, cwd=str(tmp_path), env=env)
    assert r.returncode == 0
    assert r.stdout.strip(), "a relative file_path into main must be denied"
    assert json.loads(r.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_malformed_payload_fails_open():
    """A guard bug must never brick editing -- it declines to have an opinion."""
    r = subprocess.run([sys.executable, str(HOOK)], input="not json",
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert not r.stdout.strip()


def test_settings_wires_bash_into_the_matcher():
    """The script can be perfect and still never run.

    The whole bypass was a matcher that listed three tools and not Bash, so the
    guard was never invoked for the shape that did the damage. Nothing else
    checks that the two halves agree.
    """
    # This checkout's settings, not MAIN's: in a worktree they are different
    # files, and a change ships with the checkout that holds it.
    repo = HOOK.parent.parent.parent
    settings = json.loads((repo / ".claude/settings.json").read_text())
    matchers = [
        entry.get("matcher", "")
        for entry in settings.get("hooks", {}).get("PreToolUse", [])
        if "protect-main.py" in json.dumps(entry)
    ]
    assert matchers, "no PreToolUse entry invokes protect-main.py"
    for matcher in matchers:
        tools = set(matcher.split("|"))
        missing = {"Write", "Edit", "NotebookEdit", "Bash"} - tools
        assert not missing, f"protect-main.py is not wired for {sorted(missing)}"
