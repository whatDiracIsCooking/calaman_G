#!/usr/bin/env bash
# Report -- and bump -- the Claude Code version the GPU images are pinned to.
#
#   devtools/claude-version.sh                 pinned vs registry vs installed
#   devtools/claude-version.sh --apply         rewrite the pin to the registry's latest
#   devtools/claude-version.sh --apply 2.1.300 rewrite it to an explicit version
#   devtools/claude-version.sh --apply --rebuild   ... and rebuild the images
#                                                  that already exist locally
#   devtools/claude-version.sh --porcelain     one `key value` per line
#
# The pin is `ARG CLAUDE_CODE_VERSION=` in each file named by CLAUDE_PIN_FILES
# in devtools/config.sh -- the three GPU Dockerfiles, which each install
# docker/install-claude-code.sh as their last layer. That script's header has
# why the pin exists; this one keeps the copies in step, because a bump that
# reaches two files out of three is exactly the drift it was meant to end.
#
# Exit status: 0 when every file agrees with every other (and, for --apply,
# when the rewrite landed), 1 otherwise. Being BEHIND the registry is not a
# failure -- a pin is a deliberate choice of version, and this script's job is
# to make the gap visible, not to decide it.
#:
# Reading the registry needs the network. It is one unauthenticated GET of
# https://registry.npmjs.org/<pkg>/latest with a short timeout; when it fails,
# the registry column reads `-` and everything else still works, which is what
# keeps doctor.sh usable on a box with no outbound route.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

usage() { usage_from_header "${BASH_SOURCE[0]}"; }

PKG=${CLAUDE_NPM_PACKAGE:-@anthropic-ai/claude-code}
TIMEOUT=${CLAUDE_REGISTRY_TIMEOUT:-8}

# --- the pin ---------------------------------------------------------------

# Every file that carries the ARG, as an array. A path that does not exist is
# a hard error rather than a skip: a renamed Dockerfile that silently drops
# out of this list is the drift, not a tidying-up.
pin_files() {
  config_lines "${CLAUDE_PIN_FILES:-}"
  PIN_FILES=("${CONFIG_LINES[@]}")
  local f
  for f in "${PIN_FILES[@]}"; do
    [ -f "$f" ] || { echo "claude-version: no such file: $f (CLAUDE_PIN_FILES)" >&2; exit 1; }
  done
  [ "${#PIN_FILES[@]}" -gt 0 ] || {
    echo "claude-version: CLAUDE_PIN_FILES is empty -- nothing to report on" >&2
    exit 1
  }
}

pin_in() { sed -n 's/^ARG CLAUDE_CODE_VERSION=//p' "$1" | head -1; }

# The pinned version, or nothing when the files disagree. DISAGREE is left
# holding a `file=version` list for EVERY file either way -- the whole list,
# not just up to the first clash, because a partial bump that reached one file
# of three is the case this exists to name, and naming two of them would send
# you back to grep for the third.
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

# --- the other two numbers --------------------------------------------------

registry_version() {
  local url="https://registry.npmjs.org/${PKG//\//%2F}/latest"
  curl -fsSL --max-time "$TIMEOUT" "$url" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["version"])' 2>/dev/null \
    || true
}

# What `claude` on THIS side actually is. Host and container both answer, and
# the answer means different things: in a container it should equal the pin,
# on the host it is just whatever the user installed and is reported for
# contrast rather than checked.
installed_version() {
  command -v claude >/dev/null 2>&1 || return 0
  claude --version 2>/dev/null | cut -d' ' -f1
}

semver_ok() { [[ $1 =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; }

# --- rewrite ----------------------------------------------------------------

apply_pin() {
  local want=$1 f before
  semver_ok "$want" || { echo "claude-version: '$want' is not a version" >&2; exit 1; }
  for f in "${PIN_FILES[@]}"; do
    before=$(pin_in "$f")
    [ -n "$before" ] || { echo "claude-version: $f has no ARG CLAUDE_CODE_VERSION line" >&2; exit 1; }
    sed -i "s/^ARG CLAUDE_CODE_VERSION=.*/ARG CLAUDE_CODE_VERSION=${want}/" "$f"
    if [ "$(pin_in "$f")" != "$want" ]; then
      echo "claude-version: rewrite of $f did not take" >&2; exit 1
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
      echo "claude-version: + docker/build.sh $target"
      docker/build.sh "$target"
    else
      echo "claude-version: ${PROJECT_NAME}:${target} not built here -- skipping"
      echo "                build it with: docker/build.sh $target"
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
    -*)          echo "claude-version: unknown option $1" >&2; usage >&2; exit 2 ;;
    *)           want=$1 ;;
  esac
  shift
done

pin_files

case $mode in
  porcelain)
    pinned_version || true
    printf 'pinned %s\n'    "${PINNED:--}"
    printf 'files %s\n'     "${DISAGREE[*]}"
    printf 'registry %s\n'  "$(registry_version || true)"
    printf 'installed %s\n' "$(installed_version || true)"
    [ -n "$PINNED" ]
    ;;

  report)
    latest=$(registry_version)
    here=$(installed_version)
    if pinned_version; then
      printf 'pinned     %s   (%s)\n' "$PINNED" "${PIN_FILES[*]}"
    else
      printf 'pinned     DISAGREE: %s\n' "${DISAGREE[*]}"
      echo   '           fix with: devtools/claude-version.sh --apply <version>'
    fi
    printf 'registry   %s   (%s)\n' "${latest:-unreachable}" "$PKG"
    printf 'installed  %s   (%s)\n' "${here:-none}" "$(command -v claude || echo 'not on PATH here')"
    echo
    if [ -n "$PINNED" ] && [ -n "$latest" ] && [ "$PINNED" != "$latest" ]; then
      echo "the pin is behind the registry: $PINNED -> $latest"
      echo "  bump it:  devtools/claude-version.sh --apply"
      echo "  then:     docker/build.sh cuda   (and hip / combined as you use them)"
    elif [ -n "$PINNED" ] && [ "$PINNED" = "$latest" ]; then
      echo "the pin is current"
    fi
    [ -n "$PINNED" ]
    ;;

  apply)
    if [ -z "$want" ]; then
      want=$(registry_version)
      [ -n "$want" ] || { echo "claude-version: registry unreachable and no version given" >&2; exit 1; }
      echo "claude-version: registry latest is $want"
    fi
    apply_pin "$want"
    if [ -n "$rebuild" ]; then
      rebuild_built_images
    else
      echo "claude-version: rebuild to pick it up -- docker/build.sh cuda (and hip / combined)"
      echo "                or re-run with --rebuild to do the ones already built here"
    fi
    ;;
esac
