#!/usr/bin/env bash
# Install Node and Claude Code AT AN EXPLICIT VERSION.
#
# Called by docker/Dockerfile.{cuda,hip,combined} as their last root layer --
# the third script, after install-rocm.sh and install-rocm-ds.sh, that more
# than one Dockerfile drives. Deliberately NOT in Dockerfile.base, and that is
# the whole point: base is the parent of every GPU image, so a bump there would
# invalidate the CUDA toolkit, the ~19GB ROCm install and the cuGraph wheels
# downstream of it. Pinned at the leaves, a bump re-runs this script and the
# two trivial layers after it, and nothing else.
#
# WHY THIS EXISTS AT ALL -- the drift it ends
# This replaces ghcr.io/anthropics/devcontainer-features/claude-code, which the
# three devcontainer.json files used to declare. That feature's install.sh runs
# a bare, UNPINNED `npm install -g @anthropic-ai/claude-code`, so the version
# baked into an image is "whatever the registry had when that layer was first
# built". Three things then freeze it there:
#
#   1. the feature's content never changes, so its layer is a permanent cache
#      hit -- a rebuild re-uses it and installs nothing;
#   2. `devtools/devcontainer.sh rebuild` passes --remove-existing-container,
#      which recreates the CONTAINER from the same IMAGE, bumps nothing, and
#      reports success;
#   3. .devcontainer/seed-claude-config.sh pins autoUpdates false -- correctly,
#      since the npm prefix is /usr and root-owned -- so claude cannot
#      self-correct either.
#
# devcontainer-lock.json does not help: it pins the FEATURE's digest, not the
# version of the package the feature installs. Measured on the reference box
# before this landed: four images off one lockfile, three at claude 2.1.197 and
# one at 2.1.289 -- the version was a function of build date, and nothing
# reported it. The pin is a reviewable line in git instead (ARG
# CLAUDE_CODE_VERSION in each GPU Dockerfile), bumped by
# devtools/claude-version.sh and reported by devtools/doctor.sh.
set -euo pipefail

: "${CLAUDE_CODE_VERSION:?set by the ARG of the same name in the Dockerfile}"

# Node is here only because Claude Code is an npm package -- nothing else in
# this tree needs it (the `npx` in devtools/devcontainer.sh runs on the HOST).
# The feature installed Node 18, which is EOL; 22 is the current LTS.
NODE_MAJOR=${NODE_MAJOR:-22}

install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
  | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] \
https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
  > /etc/apt/sources.list.d/nodesource.list
apt-get update
apt-get install -y --no-install-recommends nodejs
rm -rf /var/lib/apt/lists/*

# The nodesource deb sets prefix=/usr, so claude lands at /usr/bin/claude --
# exactly where the feature used to put it, which is what keeps
# seed-claude-config.sh's note about installMethod describing this image.
prefix=$(npm prefix -g)
if [ "$prefix" != /usr ]; then
  echo "install-claude-code: npm global prefix is '$prefix', expected /usr" >&2
  echo "  seed-claude-config.sh and doctor.sh both assume /usr/bin/claude" >&2
  exit 1
fi

npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"

# The guard that matters. npm turns a missing or yanked version into an error
# by itself, but a TYPO'd dist-tag resolves to something else and installs it
# quietly -- and a silently-wrong Claude in every container is the exact
# failure this script exists to end. Fail the build instead.
installed=$(claude --version | cut -d' ' -f1)
if [ "$installed" != "$CLAUDE_CODE_VERSION" ]; then
  echo "install-claude-code: asked for ${CLAUDE_CODE_VERSION}, got ${installed}" >&2
  exit 1
fi

echo "=== Claude Code ${CLAUDE_CODE_VERSION} at $(command -v claude), node $(node --version) ==="
