#!/usr/bin/env bash
# Report -- and bump -- the Antigravity CLI version the GPU images are pinned to.
#
#   devtools/agy-version.sh                 pinned vs upstream vs installed
#   devtools/agy-version.sh --apply         rewrite the pin to upstream's latest
#   devtools/agy-version.sh --apply 1.3.3   ... to an explicit version (see below)
#   devtools/agy-version.sh --apply --rebuild   ... and rebuild the images
#                                               that already exist locally
#   devtools/agy-version.sh --porcelain     one `key value` per line
#
# The claude-version.sh of the second agent, and deliberately its twin: same
# three modes, same output shape, same "behind is not a failure" rule. Read
# that script first if you are changing both.
#
# THE PIN IS IN TWO FILES, NOT ONE, and that is the only real difference.
# `ARG AGY_VERSION=` lives in each file named by AGY_PIN_FILES (the three GPU
# Dockerfiles); the artifact coordinate it names -- an immutable GCS object URL
# plus a sha512, per architecture -- lives in AGY_PIN_MANIFEST
# (docker/agy-pin.json). docker/install-agy.sh explains why: npm resolves a
# version to an artifact for claude, and agy has no equivalent index, so the
# artifact has to be pinned by hand. Both halves are checked here, and
# `--apply` always rewrites them together.
#
# WHY `--apply <version>` IS NARROWER THAN claude's. Upstream publishes a
# LATEST-ONLY manifest; per-version URLs 404. So the coordinate for an
# arbitrary older version is simply not obtainable, and this script will not
# rewrite the ARG to a version whose artifact it cannot also pin -- that would
# produce exactly the half-applied bump install-agy.sh fails the build over.
# An explicit version is therefore accepted only when it IS upstream's current
# one (which makes the gesture safe rather than useless); to go backwards, edit
# docker/agy-pin.json by hand with a coordinate you have, then re-run --apply
# with that version.
#
# Exit status: 0 when every copy agrees with every other (and, for --apply,
# when the rewrite landed), 1 otherwise. Being BEHIND upstream is not a
# failure -- a pin is a deliberate choice of version, and this script's job is
# to make the gap visible, not to decide it.
#:
# Reading upstream needs the network: two unauthenticated GETs of
# $AGY_MANIFEST_URL/linux_{amd64,arm64}.json with a short timeout. When they
# fail, the upstream column reads `-` and everything else still works, which is
# what keeps doctor.sh usable on a box with no outbound route.
#
# WHAT THE COORDINATE CHECK DOES NOT PROVE. It checks the pin file's SHAPE --
# that a version is there, that both architectures have a url and a
# 128-character sha512, and that each url carries the version it claims. It
# does NOT check the digest against the artifact, because that means
# downloading ~62MB per architecture, and this runs on the path of every
# `doctor.sh` and every `devcontainer.sh up`. So a well-formed but WRONG digest
# reports "the pin is current" here and fails in docker/install-agy.sh instead.
# That is the right place for it: the build is where the bytes are fetched, and
# it is the one caller that has them to compare.
#
# NOTE what `installed` does and does not mean here. Unlike claude -- which
# npm-installs into root-owned /usr and therefore cannot move -- agy is
# installed USER-OWNED at ~/.local/bin/agy and self-updates on a 15-minute
# check with no opt-out (docker/install-agy.sh has the detail). So `installed`
# drifting above `pinned` inside a container is EXPECTED, not a fault: the pin
# says what the image shipped. That is why this script reports the three
# numbers rather than asserting any two of them are equal.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

usage() { usage_from_header "${BASH_SOURCE[0]}"; }

MANIFEST_URL=${AGY_MANIFEST_URL:?set in devtools/config.sh}
PIN_MANIFEST=${AGY_PIN_MANIFEST:?set in devtools/config.sh}
TIMEOUT=${AGY_REGISTRY_TIMEOUT:-8}

# The architectures the pin covers: dpkg spelling (the key in agy-pin.json,
# and what docker/install-agy.sh looks itself up by) to upstream's platform
# spelling (the manifest filename). Both are pinned because either image can
# be built on either, and a coordinate for only one arch turns the other into a
# build failure at the sha512 step rather than a clear "not pinned here".
ARCHES="amd64:linux_amd64 arm64:linux_arm64"

# --- the pin, half one: the ARG ---------------------------------------------

# Every file that carries the ARG, as an array. A path that does not exist is
# a hard error rather than a skip, exactly as in claude-version.sh: a renamed
# Dockerfile that silently drops out of this list is the drift, not a tidy-up.
pin_files() {
  config_lines "${AGY_PIN_FILES:-}"
  PIN_FILES=("${CONFIG_LINES[@]}")
  local f
  for f in "${PIN_FILES[@]}"; do
    [ -f "$f" ] || { echo "agy-version: no such file: $f (AGY_PIN_FILES)" >&2; exit 1; }
  done
  [ "${#PIN_FILES[@]}" -gt 0 ] || {
    echo "agy-version: AGY_PIN_FILES is empty -- nothing to report on" >&2
    exit 1
  }
}

pin_in() { sed -n 's/^ARG AGY_VERSION=//p' "$1" | head -1; }

# The pinned version, or nothing when the files disagree. DISAGREE is left
# holding a `file=version` list for EVERY file either way -- the whole list,
# not just up to the first clash, so a partial bump names all three rather
# than sending you back to grep for the third.
pinned_version() {
  local f v first=""
  DISAGREE=()
  PINNED=""
  for f in "${PIN_FILES[@]}"; do
    v=$(pin_in "$f")
    DISAGREE+=("$f=${v:-NONE}")
    [ -n "$first" ] || first=$v
    [ "$v" = "$first" ] || first=DISAGREE
  done
  [ "$first" = DISAGREE ] && return 1
  PINNED=$first
  [ -n "$PINNED" ]
}

# --- the pin, half two: the artifact coordinate -----------------------------

# The version agy-pin.json claims, or nothing when the file is missing or
# unreadable as JSON. This is the number install-agy.sh cross-checks the ARG
# against, so a mismatch here is the half-applied bump that fails a build.
manifest_pin_version() {
  [ -r "$PIN_MANIFEST" ] || return 0
  jq -r '.version // empty' "$PIN_MANIFEST" 2>/dev/null || true
}

# Every problem with the pin file, one per line, or nothing when it is sound.
# Checked rather than assumed because this file is the half a hand-edit is
# most likely to reach: a dropped architecture or a 64-character "sha512"
# otherwise surfaces as a build failure deep in a 40GB image.
manifest_pin_problems() {
  local arch plat url sha v
  if [ ! -r "$PIN_MANIFEST" ]; then
    echo "missing: $PIN_MANIFEST"
    return 0
  fi
  if ! jq -e . "$PIN_MANIFEST" >/dev/null 2>&1; then
    echo "not valid JSON: $PIN_MANIFEST"
    return 0
  fi
  v=$(manifest_pin_version)
  [ -n "$v" ] || echo "no .version in $PIN_MANIFEST"
  for arch in $ARCHES; do
    arch=${arch%%:*}
    url=$(jq -r --arg a "$arch" '.platforms[$a].url // empty' "$PIN_MANIFEST")
    sha=$(jq -r --arg a "$arch" '.platforms[$a].sha512 // empty' "$PIN_MANIFEST")
    [ -n "$url" ] || { echo "no url for $arch"; continue; }
    [ -n "$sha" ] || echo "no sha512 for $arch"
    [ "${#sha}" -eq 128 ] || echo "sha512 for $arch is ${#sha} chars, expected 128"
    # The URL embeds <version>-<build id>; a version that is not in its own
    # URL means the two were bumped out of step.
    case "$url" in
      *"/antigravity-cli/$v-"*) ;;
      *) echo "url for $arch does not carry version $v" ;;
    esac
  done
}

# --- the other two numbers --------------------------------------------------

# Upstream's current version. Only amd64 is consulted: the manifests are cut
# per release and have never disagreed, and --apply re-reads both anyway.
upstream_version() {
  curl -fsSL --max-time "$TIMEOUT" "$MANIFEST_URL/linux_amd64.json" 2>/dev/null \
    | jq -r '.version // empty' 2>/dev/null \
    || true
}

# What `agy` on THIS side actually is. Host and container both answer, and the
# answer means different things: on the host it is whatever you installed, and
# in a container it is the pin PLUS however far the self-updater has carried it
# since the image was built. Reported for contrast in both cases -- see the
# note in the header about why neither is asserted.
installed_version() {
  command -v agy >/dev/null 2>&1 || return 0
  agy --version 2>/dev/null | head -1
}

semver_ok() { [[ $1 =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; }

# --- rewrite ----------------------------------------------------------------

# Rebuild AGY_PIN_MANIFEST from upstream, for every architecture in ARCHES.
# Fails without touching the file unless all of them answer: a pin file holding
# a fresh amd64 coordinate beside a stale arm64 one is worse than no rewrite,
# because install-agy.sh would happily install the stale half on an arm box.
apply_manifest() {
  local want=$1 arch plat json v url sha tmp
  tmp=$(mktemp)
  printf '{\n  "version": "%s",\n  "platforms": {\n' "$want" >"$tmp"
  local first=1
  for arch in $ARCHES; do
    plat=${arch##*:}
    arch=${arch%%:*}
    json=$(curl -fsSL --max-time "$TIMEOUT" "$MANIFEST_URL/$plat.json" 2>/dev/null || true)
    [ -n "$json" ] || { rm -f "$tmp"; echo "agy-version: could not fetch $MANIFEST_URL/$plat.json" >&2; exit 1; }
    v=$(printf '%s' "$json" | jq -r '.version // empty')
    url=$(printf '%s' "$json" | jq -r '.url // empty')
    sha=$(printf '%s' "$json" | jq -r '.sha512 // empty')
    if [ "$v" != "$want" ]; then
      rm -f "$tmp"
      echo "agy-version: $plat is at $v, not the $want being applied" >&2
      echo "  upstream publishes only its latest; see this script's header" >&2
      exit 1
    fi
    [ -n "$url" ] && [ -n "$sha" ] || { rm -f "$tmp"; echo "agy-version: $plat manifest has no url/sha512" >&2; exit 1; }
    [ "$first" = 1 ] || printf ',\n' >>"$tmp"
    first=0
    printf '    "%s": {\n      "url": "%s",\n      "sha512": "%s"\n    }' "$arch" "$url" "$sha" >>"$tmp"
  done
  printf '\n  }\n}\n' >>"$tmp"
  jq -e . "$tmp" >/dev/null || { rm -f "$tmp"; echo "agy-version: produced invalid JSON" >&2; exit 1; }
  mv "$tmp" "$PIN_MANIFEST"
  local listed=""
  for arch in $ARCHES; do listed="${listed:+$listed, }${arch%%:*}"; done
  printf '  %s: coordinates for %s -> %s\n' "$PIN_MANIFEST" "$listed" "$want"
}

apply_pin() {
  local want=$1 f before
  semver_ok "$want" || { echo "agy-version: '$want' is not a version" >&2; exit 1; }
  # The coordinate first: if upstream cannot supply it, the ARGs are left
  # alone rather than rewritten to a version nothing can install.
  apply_manifest "$want"
  for f in "${PIN_FILES[@]}"; do
    before=$(pin_in "$f")
    [ -n "$before" ] || { echo "agy-version: $f has no ARG AGY_VERSION line" >&2; exit 1; }
    sed -i "s/^ARG AGY_VERSION=.*/ARG AGY_VERSION=${want}/" "$f"
    if [ "$(pin_in "$f")" != "$want" ]; then
      echo "agy-version: rewrite of $f did not take" >&2; exit 1
    fi
    printf '  %s: %s -> %s\n' "$f" "$before" "$want"
  done
}

# Rebuild only what is already built. Nothing here invents a ~40GB `combined`
# build on a box that has never had one; the images it skips are named, so the
# choice stays the caller's.
rebuild_built_images() {
  local f target
  for f in "${PIN_FILES[@]}"; do
    target=${f##*Dockerfile.}
    if docker image inspect "${PROJECT_NAME}:${target}" >/dev/null 2>&1; then
      echo "agy-version: + docker/build.sh $target"
      docker/build.sh "$target"
    else
      echo "agy-version: ${PROJECT_NAME}:${target} not built here -- skipping"
      echo "             build it with: docker/build.sh $target"
    fi
  done
}

# --- arguments --------------------------------------------------------------

mode=report
want=""
rebuild=
while [ $# -gt 0 ]; do
  case $1 in
    --apply)     mode=apply ;;
    --rebuild)   mode=apply; rebuild=1 ;;
    --porcelain) mode=porcelain ;;
    -h|--help)   usage; exit 0 ;;
    -*)          echo "agy-version: unknown option $1" >&2; usage >&2; exit 2 ;;
    *)           want=$1 ;;
  esac
  shift
done

pin_files

case $mode in
  porcelain)
    pinned_version || true
    problems=$(manifest_pin_problems)
    printf 'pinned %s\n'    "${PINNED:--}"
    printf 'files %s\n'     "${DISAGREE[*]}"
    printf 'coordinate %s\n' "$(manifest_pin_version || true)"
    printf 'coordinate_problems %s\n' "$(printf '%s' "$problems" | tr '\n' ';')"
    printf 'upstream %s\n'  "$(upstream_version || true)"
    printf 'installed %s\n' "$(installed_version || true)"
    [ -n "$PINNED" ] && [ -z "$problems" ]
    ;;

  report)
    latest=$(upstream_version)
    here=$(installed_version)
    coord=$(manifest_pin_version)
    problems=$(manifest_pin_problems)
    if pinned_version; then
      printf 'pinned      %s   (%s)\n' "$PINNED" "${PIN_FILES[*]}"
    else
      printf 'pinned      DISAGREE: %s\n' "${DISAGREE[*]}"
      echo   '            fix with: devtools/agy-version.sh --apply'
    fi
    printf 'coordinate  %s   (%s)\n' "${coord:--}" "$PIN_MANIFEST"
    printf 'upstream    %s   (%s)\n' "${latest:-unreachable}" "$MANIFEST_URL"
    printf 'installed   %s   (%s)\n' "${here:-none}" "$(command -v agy || echo 'not on PATH here')"
    echo
    if [ -n "$problems" ]; then
      echo "the artifact coordinate is unusable:"
      printf '  %s\n' "$problems"
      echo "  fix: devtools/agy-version.sh --apply"
    elif [ -n "$PINNED" ] && [ -n "$coord" ] && [ "$PINNED" != "$coord" ]; then
      echo "the two halves of the pin disagree: ARG says $PINNED, $PIN_MANIFEST says $coord"
      echo "  a build would fail in docker/install-agy.sh rather than install either"
      echo "  fix: devtools/agy-version.sh --apply"
    elif [ -n "$PINNED" ] && [ -n "$latest" ] && [ "$PINNED" != "$latest" ]; then
      echo "the pin is behind upstream: $PINNED -> $latest"
      echo "  bump it:  devtools/agy-version.sh --apply"
      echo "  then:     docker/build.sh cuda   (and hip / combined as you use them)"
    elif [ -n "$PINNED" ] && [ "$PINNED" = "$latest" ]; then
      echo "the pin is current"
    fi
    [ -n "$PINNED" ] && [ -z "$problems" ]
    ;;

  apply)
    latest=$(upstream_version)
    if [ -z "$want" ]; then
      want=$latest
      [ -n "$want" ] || { echo "agy-version: upstream unreachable and no version given" >&2; exit 1; }
      echo "agy-version: upstream latest is $want"
    elif [ -n "$latest" ] && [ "$want" != "$latest" ]; then
      # The narrowing promised in the header, refused loudly rather than
      # half-applied.
      echo "agy-version: cannot pin $want -- upstream publishes only $latest" >&2
      echo "  there is no per-version manifest, so the artifact URL and sha512" >&2
      echo "  for $want are not obtainable. To go backwards, put a coordinate" >&2
      echo "  you already have into $PIN_MANIFEST by hand, then re-run with" >&2
      echo "  that version. See this script's header." >&2
      exit 1
    fi
    apply_pin "$want"
    if [ -n "$rebuild" ]; then
      rebuild_built_images
    else
      echo "agy-version: rebuild to pick it up -- docker/build.sh cuda (and hip / combined)"
      echo "             or re-run with --rebuild to do the ones already built here"
    fi
    ;;
esac
