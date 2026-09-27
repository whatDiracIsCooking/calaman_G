#!/usr/bin/env python3
"""PreToolUse guard: block edits to the primary (`main`) checkout of THIS repo.

The failure mode this defends against: a session told to work in a
`.claude/worktrees/<name>` worktree strays back into the shared main checkout
and edits it directly. It denies file-mutating tools whose target resolves
under the primary worktree, with two carve-outs:

  * the nested `<main>/.claude/worktrees/` subtree is ALLOWED (that IS where
    the lightweight worktrees live), and
  * anything outside the primary worktree is ALLOWED -- other repos, the
    sibling `...-root/<name>` container worktrees, /tmp, etc.

The primary worktree is found dynamically as the parent of the shared git
common dir (`git rev-parse --git-common-dir`), so there is no hardcoded path
and every checkout of this repo shares the same target.

TWO TOOL SHAPES, because guarding only the first one is a hole wide enough to
drive the whole session through:

  Write/Edit/NotebookEdit  one declared path in `file_path`/`notebook_path`.
                           Exact: the path either resolves under main or not.

  Bash                     an arbitrary command string. A session that edits
                           with `sed -i`, a `>` redirect or a python heredoc
                           mutates the primary checkout just as thoroughly and
                           used to sail straight past this guard, which matched
                           only the three tools above. Worse, it is the shape an
                           agent reaches for when told to prefer shell tools --
                           so the bypass is the DEFAULT path, not an exotic one.
                           Necessarily a heuristic; see _bash_targets.

Every terminal decision appends one tab-separated line to
<main_root>/.logs/protect-main.log (git-ignored): timestamp, decision
(allow/deny/bypass), session id, tool, cwd, detail. It exists because the guard
was otherwise INVISIBLE -- when an edit reached the primary checkout there was
no way to tell whether the guard ran and allowed, never ran, or was waved
through by the escape hatch, and those three have very different fixes. The
session id is the forensic key: it names which concurrent session did it. Set
$PROTECT_MAIN_LOG_DIR to send the log elsewhere (the tests point it at a tmp
dir so a run does not spam the real one). Logging never raises.

Escape hatch: export CLAUDE_ALLOW_MAIN_EDITS=1 to edit main deliberately. A
bypass is still LOGGED (as `bypass`, with what it let through), so a leaked or
forgotten hatch is catchable rather than silent.

Wired from the committed `.claude/settings.json` as a PreToolUse hook on
Write|Edit|NotebookEdit|Bash. Fails OPEN on any error: a guard bug must never
brick editing.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime

# Commands whose whole job is to change files. Matched only in COMMAND position
# (first word of a pipeline segment), so `grep "rm -rf" notes.txt` is not a
# mutation just for containing the word.
MUTATING = {
    "rm", "rmdir", "mv", "cp", "install", "truncate", "dd", "shred", "ln",
    "mkdir", "touch", "chmod", "chown", "patch", "tee", "sed", "awk", "perl",
    "python", "python3", "ruby", "node", "gzip", "gunzip", "tar", "unzip",
}

# `sed`/`awk` only mutate in place; everything else they do is a read.
IN_PLACE = re.compile(r"(?:^|\s)(?:-i\b|--in-place\b)")

# Archive tools write only in some modes. `tar -tzf a.tgz` LISTS an archive and
# was being denied as if it unpacked one. tar is judged by what it was asked to
# do (extract/create); the others by their read-only flags.
ARCHIVE_WRITE_MODES = {
    "tar": ("x", "c"),      # short clusters: -xzf writes, -tzf does not
}
ARCHIVE_READ_FLAGS = {
    "unzip": ("-l", "-t", "-p", "-v"),
    "gzip": ("-l", "-t", "--list", "--test"),
    "gunzip": ("-l", "-t", "--list", "--test"),
}

# Interpreters mutate only when the script body says so. Checked against the
# whole segment, which is what catches a `python3 - <<EOF` heredoc.
SCRIPT_WRITES = re.compile(
    r"write_text\(|writelines\(|open\([^)]*['\"][wax]|"
    r"shutil\.(?:copy|move|rmtree)|os\.(?:remove|rename|replace|unlink|mkdir)|"
    r"\.unlink\(|\.rename\(|\.mkdir\(|\.touch\("
)
INTERPRETERS = {"python", "python3", "perl", "ruby", "node", "awk"}

# `cmd SRC... DEST` -- only the final operand is written.
COPY_LIKE = {"cp", "mv", "ln", "install"}

# git subcommands that rewrite the working tree (as opposed to the index/refs).
GIT_MUTATING = {"apply", "rm", "mv", "restore", "clean", "stash", "checkout"}

# git operations judged by what they DO to the shared checkout rather than by
# which path they name, so the path-based pass above cannot see them.
#
#   reset --hard   throws away uncommitted work in a tree other people share,
#                  and names no path to catch it by.
#   commit         lands work on whatever branch the primary checkout is on.
#                  Guarded ONLY when that is `main`: the repo ships through
#                  PRs, and the pr skill's "never commit on main" was an
#                  instruction with nothing enforcing it. A topic branch
#                  checked out there is ordinary and stays allowed.
#
# `git add` is deliberately NOT here. Staging is inert and one `git restore
# --staged` undoes it, so blocking it buys nothing and costs a prompt every
# time someone stages in main. Nor is `git push`: a local hook cannot bind
# other machines or other clients, and branch protection is the real answer --
# pretending otherwise would be false confidence.
GIT_RESET_DESTRUCTIVE = ("--hard", "--merge")

# Redirects (`>`, `>>`) are found by _redirect_targets, which tracks quote state
# so a `>` inside a quoted argument is not read as one. A regex could not: it
# matched any `>`, so `echo "a > b"` and `git log --grep "x -> y"` looked like
# writes into files `b` / `y`. Comments are removed up front by _strip_comments
# for the same reason -- an arrow in prose (`squash -> SHAs`) is not a redirect.

# Word-boundary characters that let a following `#` begin a shell comment (plus
# start-of-string and whitespace). A `#` mid-word or in quotes is not a comment.
COMMENT_STARTERS = frozenset(";&|()")

# Characters that terminate a bare (unquoted) shell token.
TOKEN_ENDERS = frozenset(" \t;|&<>()")

# Quoted strings, which is where a heredoc's paths live.
QUOTED = re.compile(r"""['"]([^'"]{2,})['"]""")

# Pipeline/list separators. Splitting on these gives us command position.
SEGMENT = re.compile(r"(?:\|\||&&|[;|\n()])")

# Targets that are never the repo, however the command spells them.
HARMLESS = ("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty")


# Diagnostic log context, filled in main() as soon as each field is known. _log
# reads it so the terminal-decision helpers below need no extra plumbing.
_LOGCTX: dict[str, str] = {}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _log(decision: str, detail: str = "") -> None:
    """Append one decision line. Never raises -- logging must not brick the guard.

    Writes to $PROTECT_MAIN_LOG_DIR when set, else <main_root>/.logs. With
    neither known yet (git unavailable, or a pre-parse exit) there is nowhere to
    write, so the line is dropped rather than guessed at."""
    override = os.environ.get("PROTECT_MAIN_LOG_DIR")
    root = _LOGCTX.get("main_root")
    if not override and not root:
        return
    try:
        logdir = override or os.path.join(root, ".logs")
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "protect-main.log"), "a") as f:
            f.write("\t".join((
                _now(), decision,
                _LOGCTX.get("session", "-"),
                _LOGCTX.get("tool", "-"),
                _LOGCTX.get("cwd", "-"),
                " ".join(detail.split())[:400],
            )) + "\n")
    except Exception:
        pass


def _allow(detail: str = "") -> None:
    # No output + exit 0 == no opinion; the tool proceeds.
    _log("allow", detail)
    sys.exit(0)


def _deny(reason: str) -> None:
    # The escape hatch is honoured HERE, not at the top of main(), so a bypass is
    # logged WITH the thing it let through -- a leaked CLAUDE_ALLOW_MAIN_EDITS is
    # exactly what this log exists to catch, and "bypass" with no target is half a
    # clue. The cost is that a hatch-on session runs the full analysis before
    # allowing; that is a rare, deliberate session and the analysis is cheap.
    if os.environ.get("CLAUDE_ALLOW_MAIN_EDITS"):
        _log("bypass", reason)
        sys.exit(0)
    _log("deny", reason)
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def _primary_worktree(start: str) -> str:
    """Parent of the shared git common dir == the primary (main) worktree."""
    common = subprocess.check_output(
        ["git", "-C", start, "rev-parse",
         "--path-format=absolute", "--git-common-dir"],
        stderr=subprocess.DEVNULL, text=True,
    ).strip()
    return os.path.dirname(os.path.realpath(common))


def _existing_dir(path: str, fallback: str) -> str:
    """Nearest existing ancestor of a (possibly not-yet-created) path."""
    d = os.path.dirname(path)
    while d and not os.path.isdir(d):
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return d if os.path.isdir(d) else fallback


def _words(segment: str) -> list[str]:
    """Split a segment into words, tolerating the unbalanced quotes a heredoc
    leaves behind. shlex is right when it works and useless when it raises."""
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _strip_comments(command: str) -> str:
    """Remove unquoted shell comments before analysis.

    A `#` begins a comment when it is outside quotes and at the start of a word
    -- start of the command, or right after whitespace or a `;`/`&`/`|`/`(`/`)`
    separator. The shell never runs that text, but the guard used to read it: an
    arrow in a trailing note (`squash -> SHAs differ`) matched the redirect scan
    as a `>` into a file `SHAs`, denying the teardown that carried the comment.
    A quoted or mid-word `#` (a URL fragment, ``"$#"``, ``a#b``) is not a comment
    and is left untouched, so a real mutation on the line is never stripped."""
    out: list[str] = []
    quote: str | None = None
    prev = ""
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if quote:
            out.append(c)
            if c == quote:
                quote = None
            prev = c
        elif c in ("'", '"'):
            quote = c
            out.append(c)
            prev = c
        elif c == "#" and (prev == "" or prev.isspace() or prev in COMMENT_STARTERS):
            nl = command.find("\n", i)
            if nl == -1:
                break
            i = nl  # keep the newline; the next iteration appends it
            prev = "\n"
            continue
        else:
            out.append(c)
            prev = c
        i += 1
    return "".join(out)


def _read_token(s: str, j: int) -> tuple[str, int]:
    """One shell token at index j -- a quoted string (unquoted) or a bare run up
    to whitespace/metachar -- and the index just past it."""
    n = len(s)
    if j >= n:
        return "", j
    if s[j] in ("'", '"'):
        q = s[j]
        k = j + 1
        while k < n and s[k] != q:
            k += 1
        return s[j + 1:k], (k + 1 if k < n else k)
    k = j
    while k < n and s[k] not in TOKEN_ENDERS:
        k += 1
    return s[j:k], k


def _redirect_targets(segment: str) -> list[str]:
    """Files a segment redirects into (`> f`, `>> f`), quote-aware.

    Tracks quote state so a `>` inside a quoted argument (`echo "a > b"`,
    `git log --grep "x -> y"`) is not a redirect -- the regex this replaced saw
    every `>` and denied read-only commands that merely quoted one. The target
    may itself be quoted (`> "my file"`), so it is read as a token, not a regex
    run. fd forms are not file writes: a `>` right after a digit or `&` (`2>&1`,
    `2> f`) is skipped, as is a `>&1` dup, mirroring the old scan."""
    targets: list[str] = []
    i, n = 0, len(segment)
    quote: str | None = None
    while i < n:
        c = segment[i]
        if quote:
            if c == quote:
                quote = None
            i += 1
        elif c in ("'", '"'):
            quote = c
            i += 1
        elif c == ">":
            prev = segment[i - 1] if i else ""
            if prev.isdigit() or prev == "&":  # fd redirect/dup, not a file
                i += 1
                continue
            j = i + 1
            if j < n and segment[j] == ">":  # `>>`
                j += 1
            while j < n and segment[j] in " \t":
                j += 1
            if j < n and segment[j] == "&":  # `>&1` fd dup, not a file
                i = j + 1
                continue
            tok, j = _read_token(segment, j)
            if tok:
                targets.append(tok)
            i = j
        else:
            i += 1
    return targets


def _bash_targets(command: str) -> list[str] | None:
    """Paths a shell command would write, or None if it writes nothing.

    Deliberately a heuristic -- deciding what an arbitrary shell string touches
    is undecidable, and a guard that tried to be exact would either block every
    command or none. The bias is: recognise the mutation shapes an agent
    actually uses (a redirect, `sed -i`, `tee`, an interpreter heredoc, the
    coreutils verbs, git's tree-rewriting subcommands), and collect every
    path-ish token near them. Over-collecting is safe -- a token that resolves
    outside the primary checkout is ignored, and the escape hatch covers a
    genuine false positive. Under-collecting is the failure that matters, which
    is why an unparseable mutation returns an empty list (deny on cwd) rather
    than None (allow).
    """
    targets: list[str] = []
    mutates = False

    for segment in SEGMENT.split(command):
        segment = segment.strip()
        if not segment:
            continue

        for tgt in _redirect_targets(segment):
            mutates = True
            targets.append(tgt)

        words = _words(segment)
        if not words:
            continue
        verb = os.path.basename(words[0])
        # `VAR=x cmd ...` -- step over leading assignments.
        i = 0
        while i < len(words) and re.fullmatch(r"\w+=.*", words[i]):
            i += 1
        if i < len(words):
            verb = os.path.basename(words[i])
        rest = words[i + 1:]

        if verb == "git":
            if rest and rest[0] in GIT_MUTATING:
                mutates = True
                targets.extend(rest[1:])
            continue

        if verb not in MUTATING:
            continue

        if verb in ("sed", "awk"):
            if not IN_PLACE.search(segment):
                continue  # a read: `sed -n 1,5p file`
            # `sed -i EXPR FILE...`: the script is an operand but NOT a path,
            # and `s/a/b/` resolves under the repo like any relative path would
            # -- which denied every in-place edit regardless of its real target.
            # With -e/-f the script is attached to its flag and every remaining
            # operand is a file.
            operands = [w for w in rest if not w.startswith("-")]
            if not any(w.startswith(("-e", "-f", "--expression", "--file"))
                       for w in rest):
                operands = operands[1:]
            mutates = True
            targets.extend(operands)
            continue

        if verb in ARCHIVE_WRITE_MODES:
            modes = ARCHIVE_WRITE_MODES[verb]
            flags = [w for w in rest if w.startswith("-")]
            long_write = any(
                w in ("--extract", "--create", "--get") for w in flags
            )
            short_write = any(
                any(m in w.lstrip("-") for m in modes)
                for w in flags if not w.startswith("--")
            )
            if not (long_write or short_write):
                continue  # listing or testing an archive is a read
        elif verb in ARCHIVE_READ_FLAGS:
            if any(w in ARCHIVE_READ_FLAGS[verb] for w in rest):
                continue

        if verb in INTERPRETERS:
            # Scan the WHOLE command, not this segment: `python3 - <<PY` and the
            # `Path(...).write_text(...)` it feeds are different segments once
            # newlines have been split on, so a segment-local test sees an
            # interpreter that writes nothing and waves the heredoc through.
            # That is precisely the bypass this whole change exists to close.
            if not SCRIPT_WRITES.search(command):
                continue  # `python3 -c "print(1)"` is not an edit
            targets.extend(QUOTED.findall(command))

        mutates = True
        if verb in COPY_LIKE:
            # `cp SRC... DEST` writes only DEST. Collecting the sources too
            # denied any command that merely READ a file in the protected
            # checkout on its way somewhere else, e.g. `cp x.py /tmp/y.py`.
            operands = [w for w in rest if not w.startswith("-")]
            targets.extend(operands[-1:])
        else:
            targets.extend(rest)
        targets.extend(QUOTED.findall(segment))

    if not mutates:
        return None
    # Drop flags and heredoc markers (`<<PY`), which are not paths.
    return [
        t for t in targets
        if not t.startswith("-") and not t.startswith("<") and "<<" not in t
    ]


def _is_ignored(path: str, main_root: str) -> bool:
    """Does git ignore this path? Then it is not part of the checkout's content.

    build/, build-*/, .slow-tier-reports/, .venv/ and the caches are generated,
    disposable and already untracked, so writing to them is not "editing the
    primary checkout" in any sense worth blocking -- and guarding them denied
    ordinary work there (`mkdir -p build`, a build log, a `--fresh` wipe) for no
    benefit. It is the same rule the .claude/worktrees/ carve-out below already
    encodes: that path is carved out precisely BECAUSE .gitignore lists it.

    Pattern-matched, so it works for a path that does not exist yet. The
    trailing-slash form is tried as well: this repo ignores build trees as
    `/build*/`, a DIRECTORY pattern, which git will not match against a bare
    `build` it cannot stat -- so `mkdir -p build` looked un-ignored purely
    because the directory did not exist yet.

    Any error means "not ignored", which keeps the guard on rather than off.
    """
    candidates = [path]
    # Only ask about the directory form for a name that could BE a directory.
    # `/build*/` happily matches `build.log/` too, so testing every path that
    # way would have made any root name starting with "build" writable -- a
    # hole, not a softening. A dot in the basename means treat it as a file.
    if not os.path.isfile(path) and "." not in os.path.basename(path):
        candidates.append(path.rstrip(os.sep) + os.sep)
    for candidate in candidates:
        try:
            if subprocess.run(
                ["git", "-C", main_root, "check-ignore", "-q", "--", candidate],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode == 0:
                return True
        except Exception:
            return False
    return False


def _git_invocation(words):
    """(cwd_override, subcommand, args) for a git command, else None.

    `-C <path>` matters: it moves where git operates, so `git -C /elsewhere
    reset --hard` is not this checkout's problem.
    """
    i = 0
    while i < len(words) and re.fullmatch(r"\w+=.*", words[i]):
        i += 1
    if i >= len(words) or os.path.basename(words[i]) != "git":
        return None
    override, j = None, i + 1
    while j < len(words) and words[j].startswith("-"):
        if words[j] in ("-C", "-c") and j + 1 < len(words):
            if words[j] == "-C":
                override = words[j + 1]
            j += 2
        else:
            j += 1
    if j >= len(words):
        return None
    return override, words[j], words[j + 1:]


def _current_branch(repo: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return ""


def _git_policy(command: str, cwd: str, main_root: str) -> None:
    """Deny git operations that damage the shared primary checkout."""
    worktrees = os.path.join(main_root, ".claude", "worktrees") + os.sep
    for segment in SEGMENT.split(command):
        words = _words(segment.strip())
        if not words:
            continue
        inv = _git_invocation(words)
        if not inv:
            continue
        override, sub, args = inv
        where = os.path.abspath(os.path.join(cwd, override)) if override else cwd
        where = os.path.abspath(where)
        if where != main_root and not where.startswith(main_root + os.sep):
            continue  # operating on some other tree
        if where.startswith(worktrees):
            continue  # a nested worktree is where work belongs

        if sub == "reset" and any(a in GIT_RESET_DESTRUCTIVE for a in args):
            _deny(
                f"Blocked: `git reset --hard` in the protected primary checkout "
                f"({main_root}) discards uncommitted work in a tree other "
                f"sessions share. Do it in a .claude/worktrees/<name> worktree, "
                f"or set CLAUDE_ALLOW_MAIN_EDITS=1 to mean it."
            )
        if sub == "commit":
            branch = _current_branch(main_root)
            if branch == "main":
                _deny(
                    f"Blocked: the primary checkout ({main_root}) is on `main`, "
                    f"and this repo ships through PRs. Branch first "
                    f"(`git switch -c <topic>`) or work in a "
                    f".claude/worktrees/<name> worktree. "
                    f"CLAUDE_ALLOW_MAIN_EDITS=1 overrides."
                )


def _check(path: str, main_root: str, what: str) -> None:
    """Deny if `path` lands in the protected checkout's tracked content."""
    worktrees = os.path.join(main_root, ".claude", "worktrees") + os.sep
    if path != main_root and not path.startswith(main_root + os.sep):
        return
    if path.startswith(worktrees):
        return  # inside a nested worktree -> fine
    if _is_ignored(path, main_root):
        return  # generated/disposable -> not the checkout's content
    _deny(
        f"Blocked: {what} is tracked content in the protected primary "
        f"checkout ({main_root}). Do this work in a .claude/worktrees/<name> "
        f"worktree instead. To edit main deliberately, set "
        f"CLAUDE_ALLOW_MAIN_EDITS=1 in the environment."
    )


def main() -> None:
    # Hatch is NOT short-circuited here any more; _deny honours it, so the bypass
    # is logged with its target. See _deny.
    data = json.load(sys.stdin)
    tool = data.get("tool_name") or ""
    ti = data.get("tool_input") or {}
    cwd = data.get("cwd") or os.getcwd()
    _LOGCTX.update(session=str(data.get("session_id") or "-"), tool=tool or "-",
                   cwd=cwd)

    if tool == "Bash":
        command = ti.get("command")
        if not command:
            _allow()
        # Shell comments never execute, so analysing them only invents targets
        # (`# squash -> SHAs` read as a redirect into `SHAs`). Strip them once,
        # before both the git-policy and the path passes see the command.
        command = _strip_comments(command)
        try:
            main_root = _primary_worktree(cwd)
        except Exception:
            _allow()
        _LOGCTX["main_root"] = main_root

        # Judged by operation, not by path -- these name no file to catch.
        _git_policy(command, os.path.abspath(cwd), main_root)

        targets = _bash_targets(command)
        if targets is None:
            _allow()  # read-only as far as we can tell

        for raw in targets:
            if raw in HARMLESS or raw.startswith("/dev/"):
                continue
            path = os.path.abspath(
                os.path.join(cwd, os.path.expanduser(raw))
            )
            _check(path, main_root, f"the command's target {raw}")

        # Fallback ONLY when the command named no path at all. If it named some
        # and they all resolved elsewhere -- `sed -i /tmp/x`, `git diff >
        # /dev/null` -- the read succeeded and the answer is allow; falling
        # through to judge it by cwd would deny every such command run from the
        # main checkout, which is noise that gets a guard switched off.
        if not targets:
            _check(os.path.abspath(cwd), main_root,
                   "this shell command's working directory")
        _allow()

    target = ti.get("file_path") or ti.get("notebook_path")
    if not target:
        _allow()

    raw = os.path.expanduser(target)
    if not os.path.isabs(raw):
        # A relative file_path resolves against the SESSION's cwd, not this hook
        # process's cwd -- os.path.abspath would silently use the latter, which is
        # not where the tool call is rooted, and a relative path into main would
        # then miss. Mirrors the Bash branch's os.path.join(cwd, ...).
        raw = os.path.join(cwd, raw)
    path = os.path.abspath(os.path.normpath(raw))
    start = _existing_dir(path, cwd)

    try:
        main_root = _primary_worktree(start)
    except Exception:
        _allow()  # not a git repo / git unavailable -> don't interfere
    _LOGCTX["main_root"] = main_root

    _check(path, main_root, path)
    _allow()  # outside the primary worktree entirely -> not our concern


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        # Fail open: never let a guard bug block edits everywhere.
        sys.exit(0)
