#!/usr/bin/env bash
# Install the Antigravity CLI (`agy`) AT AN EXPLICIT, CHECKSUMMED VERSION.
#
# Called by docker/Dockerfile.{cuda,hip,combined} beside install-claude-code.sh
# and for the same reason: pinned at the LEAVES, so a bump re-runs this one
# ~20s layer rather than the CUDA toolkit, the ~19GB ROCm install or the
# cuGraph wheels above it. Dockerfile.base would invalidate all three.
#
# WHY THIS DOES NOT RUN UPSTREAM'S install.sh
# https://antigravity.google/cli/install.sh has no way to ASK FOR A VERSION. It
# GETs a latest-only manifest (.../manifests/<platform>.json, giving version +
# url + sha512) and installs whatever that names; there is no per-version
# manifest endpoint -- they 404. So "pin a version" and "run the bootstrapper"
# are mutually exclusive. It would also install into the $HOME of whoever runs
# it (root, in a build) and then hand off to `agy install`, which rewrites
# shell profiles.
#
# What IS per-version addressable is the artifact: the GCS object path embeds
# <version>-<build id> and is immutable. docker/agy-pin.json holds that
# coordinate plus a sha512 per architecture, which makes this script the
# equivalent of npm's version selector in install-claude-code.sh -- with the
# integrity check npm would have done for us spelled out by hand.
#
# TWO FILES, ONE PIN, CROSS-CHECKED. AGY_VERSION comes from the Dockerfile ARG
# (three copies, like CLAUDE_CODE_VERSION); the coordinate comes from the pin
# file (one copy, both architectures). A bump that reaches one and not the
# other is the drift this whole arrangement exists to end, so disagreement
# FAILS THE BUILD below rather than quietly installing the other one's answer.
# devtools/agy-version.sh --apply rewrites both together.
#
# THE PIN IS WHAT THE IMAGE SHIPS, NOT WHAT YOU WILL BE RUNNING. agy
# self-updates in the background on a 15-minute check (`update.lock`,
# ~/.gemini/antigravity-cli/last_check.timestamp) and exposes NO setting,
# environment variable or flag to stop it -- the binary is installed
# user-owned here, so those updates succeed. That is the deliberate choice:
# the container tracks the host install instead of sitting pinned behind it,
# and `devtools/agy-version.sh` prints pinned / latest / installed so any gap
# is visible rather than silent. Contrast claude, which npm-installs into
# root-owned /usr and therefore cannot move.
set -euo pipefail

: "${AGY_VERSION:?set by the ARG of the same name in the Dockerfile}"

PIN=${AGY_PIN:-/tmp/agy-pin.json}
AGY_USER=${AGY_USER:-ubuntu}

[ -r "$PIN" ] || { echo "install-agy: no readable pin file at $PIN" >&2; exit 1; }

# The user's real home, rather than an assumed /home/ubuntu: the install target
# is ~/.local/bin, which is only the right path if it is actually that user's.
home=$(getent passwd "$AGY_USER" | cut -d: -f6)
[ -n "$home" ] || { echo "install-agy: no such user: $AGY_USER" >&2; exit 1; }
bindir="$home/.local/bin"

# The cross-check promised in the header. Both numbers are pins; neither is
# authoritative over the other, so name both and stop.
pinned=$(jq -r '.version' "$PIN")
if [ "$pinned" != "$AGY_VERSION" ]; then
  echo "install-agy: pin file says $pinned, ARG AGY_VERSION says $AGY_VERSION" >&2
  echo "  a half-applied bump; fix with devtools/agy-version.sh --apply" >&2
  exit 1
fi

# dpkg rather than the TARGETARCH build arg, so the script needs nothing passed
# in and cannot be handed an arch that disagrees with the image it is running
# in. These files are never cross-built -- every layer below this one runs the
# vendor's own apt install -- so the build platform IS the target platform.
arch=$(dpkg --print-architecture)
url=$(jq -r --arg a "$arch" '.platforms[$a].url // empty' "$PIN")
sha=$(jq -r --arg a "$arch" '.platforms[$a].sha512 // empty' "$PIN")
if [ -z "$url" ] || [ -z "$sha" ]; then
  echo "install-agy: pin file has no entry for architecture '$arch'" >&2
  echo "  it carries: $(jq -r '.platforms | keys | join(", ")' "$PIN")" >&2
  exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "=== agy ${AGY_VERSION} (${arch}) from ${url} ==="
curl -fsSL -o "$tmp/agy.tar.gz" "$url"

# The check that makes the URL a pin rather than a hope. Upstream's installer
# verifies too, but against a manifest it fetched in the same breath; here the
# expected digest is a reviewed line in git.
echo "${sha}  ${tmp}/agy.tar.gz" | sha512sum -c - >/dev/null || {
  echo "install-agy: sha512 mismatch for $url" >&2
  echo "  expected ${sha}" >&2
  echo "  got      $(sha512sum "$tmp/agy.tar.gz" | cut -d' ' -f1)" >&2
  exit 1
}

# The archive carries exactly one member, named `antigravity`; `agy` is that
# same binary under the name upstream's installer gives it. Extracting it by
# name rather than unpacking the archive wholesale means a repackaged upstream
# tarball fails here instead of scattering files into $bindir.
tar -xzf "$tmp/agy.tar.gz" -C "$tmp" antigravity

# Owned by the user, not root, and that is the whole self-update decision from
# the header expressed in two flags. Overwrites rather than skips: on
# Dockerfile.combined this script runs a second time, on top of the `cuda`
# image that already ran it.
install -d -o "$AGY_USER" -g "$AGY_USER" -m 0755 "$bindir"
install -o "$AGY_USER" -g "$AGY_USER" -m 0755 "$tmp/antigravity" "$bindir/agy"

# Verified AS THE USER, not as root: agy writes its state under $HOME, and a
# root-run check would seed /root/.gemini -- a directory nothing ever reads
# again, in an image whose only point of entry is `ubuntu`.
installed=$(runuser -u "$AGY_USER" -- "$bindir/agy" --version | head -1)
if [ "$installed" != "$AGY_VERSION" ]; then
  echo "install-agy: asked for ${AGY_VERSION}, got ${installed}" >&2
  exit 1
fi

echo "=== agy ${AGY_VERSION} at ${bindir}/agy (owner ${AGY_USER}) ==="
