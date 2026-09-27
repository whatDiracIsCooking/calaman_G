#!/usr/bin/env bash
# The devcontainer postCreateCommand, written once. All three variants
# (cuda, hip, combined) ran a byte-identical ~350-char one-liner inline in
# their devcontainer.json; JSON cannot include a shared fragment, so keeping
# them in sync by hand was the same drift trap CLAUDE.md calls out for
# PROJECT_NAME. Each devcontainer.json now just calls this script -- the same
# reasoning that already made seed-claude-config.sh a separate file.
#
# Deliberately NOT `set -e`: the original was a `;`-joined command list where a
# failure in one step did not abort the others, and this keeps that. `set -u`
# is also out, because GH_TOKEN and (on a broken git chain) H are read while
# possibly unset, exactly as the one-liner did.
set -o pipefail

# The workspace path. In the JSON this was ${localWorkspaceFolder}, the HOST
# path -- which is also where the workspace is bind-mounted INSIDE the
# container (workspaceMount targets ${localWorkspaceFolder}), so inside here it
# is simply the parent of this script's own .devcontainer/ directory. Deriving
# it from the script location reproduces ${localWorkspaceFolder} without the
# JSON having to pass it in.
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace=$(cd "$here/.." && pwd)
cd "$workspace" || exit 1

# Give the container its own Claude config so it does not come up logged out.
bash "$here/seed-claude-config.sh"

# Trust the checkout, pull submodules, and locate the shared pre-commit hook.
git config --global --add safe.directory "$workspace" &&
  git submodule update --init --recursive &&
  H=$(git rev-parse --git-common-dir)/hooks/pre-commit

# Install the pre-commit hook, but leave a HOST-installed one alone: the hook
# bakes in an absolute python, and a host install points at .venv while this
# container's would point at /opt/venv, so reinstalling here silently breaks
# committing from the host. Reinstall only when the hook is missing or is
# already this container's (/opt/venv). See CLAUDE.md.
if [ ! -e "$H" ] || grep -q /opt/venv "$H"; then
  pre-commit install -f
else
  echo "pre-commit: host-installed hook left as is"
fi

# Wire gh into git so https clones/pushes authenticate, but only with a token.
if [ -n "$GH_TOKEN" ]; then
  gh auth setup-git && git config --global url.https://github.com/.insteadOf git@github.com:
else
  echo "gh: no GH_TOKEN in env, skipping git credential setup"
fi
