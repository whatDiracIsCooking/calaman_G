#!/usr/bin/env bash
# Drive this worktree's devcontainer. There is usually no `devcontainer` binary
# on PATH, so every call goes through npx.
#
#   devtools/devcontainer.sh up          bring it up, reusing an existing one
#   devtools/devcontainer.sh rebuild     recreate from scratch
#   devtools/devcontainer.sh shell       open a bash shell inside
#   devtools/devcontainer.sh test [...]  run the test suite inside
#   devtools/devcontainer.sh down        stop and remove the container
#   devtools/devcontainer.sh down --all  ... every GPU variant of this worktree
#
# WHICH VARIANT: a flag BEFORE the command, naming a directory under
# .devcontainer/ -- so this repo's three configs are reachable as:
#
#   devtools/devcontainer.sh --hip shell        .devcontainer/hip/
#   devtools/devcontainer.sh --combined up      .devcontainer/combined/
#   devtools/devcontainer.sh --cuda rebuild     .devcontainer/cuda/ (the default)
#
# The flag must come before the command, because everything after `shell` and
# `test` is passed through to what runs inside. Use it on EVERY call in a
# session, teardown included -- `down` matches on the config file, so
# `--hip up` followed by a bare `down` takes down the CUDA container and leaves
# the HIP one running. `down --all` is the escape hatch.
#
# CPUSET / CPUS (from config.sh, or set for one call) bound the container's
# share of the host on `up` and `rebuild`:
#
#   CPUSET=0-11 devtools/devcontainer.sh up      pin to host cores 0-11
#   CPUS=8      devtools/devcontainer.sh up      cap at 8 cores' worth
#
# Unlike JOBS, which only bounds the test runner, these bound everything the
# container runs -- compiles started from a `shell` included.
#
# The three configs, and what each is:
#
#   .devcontainer/cuda/devcontainer.json      CUDA 13 (the default)
#   .devcontainer/hip/devcontainer.json       ROCm, no CUDA at all
#   .devcontainer/combined/devcontainer.json  both SDKs, ~40GB
#
# Three ways to choose, highest precedence first: the flag above, the
# DEVCONTAINER_CONFIG environment variable, and its default in config.sh (the
# CUDA variant -- the one that can actually build the C++ tree). The variable
# is still the way to point at a config that is not one of these, and the way
# to make a whole shell session use one:
#
#   export DEVCONTAINER_CONFIG=.devcontainer/hip/devcontainer.json
#
# Containers key on the workspace folder path AND the config file, so each
# worktree gets its own container per variant, and the variants of one worktree
# can be up at the same time. They SHARE that worktree's named volumes (see the
# comment in the json), so run the suite in one variant at a time. The script
# resolves the workspace from its own location, so it does the right thing
# whichever checkout you call it from, and from any cwd.
#
# Project-specific values (the test command, the worker count) come from
# devtools/config.sh -- edit that, not this.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

WORKSPACE=$REPO_ROOT
CLI=(npx -y @devcontainers/cli)

# A leading variant flag selects a config by DIRECTORY NAME under
# .devcontainer/. Resolved by convention rather than from a list, so adding a
# .devcontainer/<name>/ needs no edit here and this script stays copyable into
# the next repo -- the same convention doctor.sh globs for.
#
# A leading `--foo` that does not name an existing config is NOT silently
# treated as a variant: it falls through to the command dispatch and is
# reported as unknown, so a typo says so instead of failing later with a
# confusing path. Two variant flags in one call is an error rather than
# last-one-wins, which would make a copy-pasted command line quietly do
# something other than what it reads as.
variant_flag=
while [ $# -gt 0 ]; do
  case "$1" in
    # The name is restricted to a single plain directory name, which is what
    # makes interpolating it into a path safe: a name with slashes or dots
    # (`--../../elsewhere`) would otherwise reach outside .devcontainer/, and a
    # bare `--` would yield `.devcontainer//devcontainer.json`.
    --[A-Za-z0-9]*)
      name=${1#--}
      case "$name" in
        *[!A-Za-z0-9._-]*) break ;;
        *..*) break ;;
      esac
      candidate=$REPO_ROOT/.devcontainer/$name/devcontainer.json ;;
    *) break ;;
  esac
  [ -f "$candidate" ] || break
  if [ -n "$variant_flag" ]; then
    echo "error: two variant flags: $variant_flag and $1" >&2
    exit 2
  fi
  variant_flag=$1
  DEVCONTAINER_CONFIG=$candidate
  shift
done

# Which devcontainer.json to drive. Defaults in config.sh to the CUDA variant
# -- the one with the C++ toolchain -- and the flag above or an environment
# variable still wins for one call. A RELATIVE path is resolved against the
# repo root here, not the cwd: the CLI resolves --config relative to where you
# invoked it, so a bare `.devcontainer/cuda/devcontainer.json` would otherwise
# only work from the root. Empty would leave it to the CLI, which finds no
# top-level .devcontainer/devcontainer.json to fall back to -- but config.sh
# always sets it.
DC_CONFIG=()
if [ -n "${DEVCONTAINER_CONFIG:-}" ]; then
  case "$DEVCONTAINER_CONFIG" in
    /*) dc_config_path=$DEVCONTAINER_CONFIG ;;
    *) dc_config_path=$REPO_ROOT/$DEVCONTAINER_CONFIG ;;
  esac
  if [ ! -f "$dc_config_path" ]; then
    echo "error: DEVCONTAINER_CONFIG does not exist: $dc_config_path" >&2
    exit 1
  fi
  DC_CONFIG=(--config "$dc_config_path")
fi

usage() { usage_from_header "${BASH_SOURCE[0]}"; }

# This workspace's container ids, one per line. Empty when nothing matches.
#
# TWO labels, not one. Every GPU variant of a worktree -- cuda, hip, combined
# -- is built for the SAME workspace folder and so carries the same
# `devcontainer.local_folder`; only `devcontainer.config_file` tells them
# apart, and the CLI sets both on every container it creates. Matching on the
# folder alone would make `down` remove a sibling variant's container and
# `apply_cpu_limits` pin whichever one `head -1` happened to return.
#
#   --stopped      include exited containers -- what `down` wants, since a
#                  crashed run keeps the labels. The bare form is the running
#                  container, the only thing `docker update` can act on.
#   --any-config   every variant for this workspace, not just the active one.
container_ids() {
  local ps=(docker ps -q) any_config=
  while [ $# -gt 0 ]; do
    case $1 in
      --stopped) ps=(docker ps -aq) ;;
      --any-config) any_config=1 ;;
    esac
    shift
  done
  local filters=(--filter "label=devcontainer.local_folder=$WORKSPACE")
  # dc_config_path is set whenever DEVCONTAINER_CONFIG is, and config.sh always
  # defaults it. With neither, the CLI picks the config itself and there is no
  # path to match on, so fall back to the folder alone -- which is the old
  # behaviour, and only reachable by clearing DEVCONTAINER_CONFIG by hand.
  if [ -z "$any_config" ] && [ -n "${dc_config_path:-}" ]; then
    filters+=(--filter "label=devcontainer.config_file=$dc_config_path")
  fi
  "${ps[@]}" "${filters[@]}"
}

# Apply CPUSET/CPUS to the running container with `docker update`, then restart
# it if it holds an NVIDIA GPU. A no-op when neither is set, and a warning
# rather than an error when nothing is running: the limits are a refinement of
# `up`, not a precondition for it.
#
# `docker update` changes only the limits it is given, so a later call with
# just CPUSET leaves an earlier CPUS quota in place, and unsetting both here
# clears nothing -- it just stops applying. To get back to an unbounded
# container, `rebuild` it (a fresh container starts with no limits).
#
# THE RESTART IS LOAD-BEARING, and only for `--gpus`. `docker update`
# regenerates the container's device cgroup from HostConfig.Devices -- and for
# `--gpus all` that list is EMPTY. The ask lives in HostConfig.DeviceRequests
# and is honoured by the NVIDIA container runtime's OCI hook at container
# START, so regenerating the cgroup silently revokes GPU access while leaving
# every /dev/nvidia* node visible inside. The symptom is
# `nvidia-smi: Failed to initialize NVML: Unknown Error`, and because the
# device nodes are still there it reads like a driver problem rather than a
# permissions one. Worse here: CMAKE_CUDA_ARCHITECTURES=native then fails at
# CONFIGURE time, so a perfectly good tree looks like a broken change.
#
# Restarting re-runs the hook and re-injects the devices; the limits persist
# across it (they are in the container config, not the cgroup). The HIP
# variant needs none of this -- `--device=/dev/kfd` populates
# HostConfig.Devices, which `docker update` preserves -- so the restart is
# gated on DeviceRequests rather than applied blindly.
apply_cpu_limits() {
  [ -z "${CPUSET:-}" ] && [ -z "${CPUS:-}" ] && return 0
  local id flags=()
  # `|| true`: docker may be absent or its daemon down, and a missing CPU
  # pin is not a reason to fail a command that has already done its work.
  id=$(container_ids | head -1 || true)
  if [ -z "$id" ]; then
    echo "cpu-limit: no running container for $WORKSPACE (skipped)" >&2
    return 0
  fi
  [ -n "${CPUSET:-}" ] && flags+=(--cpuset-cpus "$CPUSET")
  [ -n "${CPUS:-}" ] && flags+=(--cpus "$CPUS")
  docker update "${flags[@]}" "$id" >/dev/null
  echo "cpu-limit: ${CPUSET:+cpuset-cpus=$CPUSET }${CPUS:+cpus=$CPUS }-> $id" >&2

  # See the comment above: only a hook-injected GPU needs this. `tr -cd` rather
  # than a bare test, so a container that died between the two calls yields an
  # empty string and not a stray newline that would read as non-zero. The
  # restart is non-fatal for the same reason the pin is: `up` has already done
  # its work by this point.
  local requests
  requests=$(docker inspect "$id" \
    --format '{{len .HostConfig.DeviceRequests}}' 2>/dev/null | tr -cd '0-9')
  if [ -n "$requests" ] && [ "$requests" != "0" ]; then
    if docker restart "$id" >/dev/null 2>&1; then
      echo "cpu-limit: restarted $id to restore its GPU device cgroup" >&2
    else
      echo "cpu-limit: could not restart $id -- its GPU may be unusable until you do" >&2
    fi
  fi
}

# The .git bind mount in devcontainer.json reads its host path from
# ${localEnv:WWR_GIT_DIR}, which this resolves and exports before the CLI
# runs -- so no machine-specific path is ever written into the tracked json.
# (The literal after the colon there is only a fallback for a direct
# `devcontainer up` / VS Code "Reopen in Container", which does not run this.)
#
# WHY A GUARD, NOT JUST AN EXPORT: an unresolved mount source makes docker's
# --mount refuse a path that does not exist, and the devcontainer CLI reports
# that by printing the ENTIRE failing `docker run` line -- which by then has
# -e GH_TOKEN=<your token> in it. So a bad mount puts a credential on the
# terminal. Resolving it here, before the CLI is invoked, is what stops that.
#
# The source is the repo's COMMON git dir: a worktree's own `.git` is a file
# pointing into <main>/.git/worktrees/<name>, so the whole common dir has to be
# visible inside at its host path (source == target) for git to work there.
require_git_dir() {
  local d
  d=$(git -C "$REPO_ROOT" rev-parse --path-format=absolute \
        --git-common-dir 2>/dev/null || true)
  if [ -z "$d" ] || [ ! -d "$d" ]; then
    echo "error: cannot resolve this repo's git dir for the container mount." >&2
    echo "       run devtools/devcontainer.sh from inside the gpumod checkout." >&2
    exit 1
  fi
  export WWR_GIT_DIR="$d"
}

# Resolve ROCM_GROUPS to numeric HOST gids and export one WWR_<NAME>_GID
# per group, which is what the `${localEnv:...}` references in the hip and
# combined runArgs read.
#
# WHY THE JSON CANNOT SIMPLY NAME THE GROUP: `--group-add render` resolves the
# name INSIDE the container, where docker/install-rocm.sh made `render` gid
# 110, while the bind-mounted /dev/kfd and /dev/dri/renderD* keep their HOST
# ownership. The container user then joins a group the devices do not grant,
# the container starts clean, and the first device call fails with
# hipErrorNoDevice -- indistinguishable from a machine with no AMD card.
# cpp-tier.sh --rocm has always resolved these with getent; this is the same
# resolution for the interactive path, off the same config.sh list.
#
# NON-FATAL, unlike cpp-tier.sh's, because each reference carries a baked-in
# default after the colon (`${localEnv:WWR_RENDER_GID:109}`). An
# unresolvable group leaves that value standing instead of passing an empty
# --group-add, and that default is also what a VS Code "Reopen in Container"
# gets -- nothing exports these for it. So the default is a last-known-good
# host GID, never a group NAME: a name there would be exactly the silent wrong
# answer this function exists to remove.
#
# The warning is gated on the ACTIVE config actually reading the variable, so
# the CUDA variant -- which passes no --group-add at all -- stays quiet on a
# host with no render group.
export_host_gids() {
  local grp gid var cfg=${dc_config_path:-}
  # Name the VARIANT in the warning: every config file is called
  # devcontainer.json, so the basename alone would not say which one.
  [ -z "$cfg" ] || cfg=$(basename "$(dirname "$cfg")")/$(basename "$cfg")
  while IFS= read -r grp; do
    [ -n "$grp" ] || continue
    var=WWR_${grp//-/_}_GID
    var=${var^^}
    # A group name that does not survive into an identifier is skipped rather
    # than exported: `export 'WWR_A B_GID=1'` fails, and under `set -e` that
    # would take down an `up` over a config typo.
    case "$var" in
      *[!A-Za-z0-9_]*) continue ;;
    esac
    # Numeric HOST gid, or empty; host_gid (lib.sh) owns the getent exit-2 trap.
    gid=$(host_gid "$grp")
    if [ -n "$gid" ]; then
      export "$var=$gid"
    elif grep -q "$var" "${dc_config_path:-/dev/null}" 2>/dev/null; then
      echo "warning: no '$grp' group on this host (getent found none), so" >&2
      echo "         $var is unset and $cfg falls back" >&2
      echo "         to the gid baked into it. If the GPU is then invisible" >&2
      echo "         in there, fix ROCM_GROUPS in config.sh or that fallback." >&2
    fi
  done <<<"${ROCM_GROUPS:-}"
}

# The CLI prints the whole `docker run` invocation on failure, GH_TOKEN and
# all. Everything below routes through this so a secret cannot reach the
# terminal; the exit status is the CLI's, not sed's.
#
# `sed -u` is load-bearing, not a micro-optimisation. sed's own stdout is the
# terminal, so without it sed is LINE-buffered: anything not terminated by a
# newline is held indefinitely. That silently swallows every partial-line
# write the CLI passes through -- pytest's progress dots, a compiler's
# carriage-returned status, and (before `shell` stopped routing through here)
# the shell prompt itself. GNU sed only; the images and the host are both
# Ubuntu.
run_cli() {
  local rc=0
  "${CLI[@]}" "$@" > >(sed -u -E 's/(GH_TOKEN|GITHUB_TOKEN)=[A-Za-z0-9_-]+/\1=***REDACTED***/g') \
    2> >(sed -u -E 's/(GH_TOKEN|GITHUB_TOKEN)=[A-Za-z0-9_-]+/\1=***REDACTED***/g' >&2) || rc=$?
  # Let the redirections drain before the caller reads the next line.
  wait
  return "$rc"
}

case "${1:-}" in
  up)
    shift
    require_git_dir
    export_host_gids
    run_cli up --workspace-folder "$WORKSPACE" "${DC_CONFIG[@]}" "$@"
    apply_cpu_limits
    ;;
  rebuild)
    # Needed after editing devcontainer.json or the Dockerfile: a plain `up`
    # reuses the running container and silently applies none of it, while
    # reporting success and the same container id. The workspace is a bind
    # mount and the caches are named volumes, so both survive; anything in the
    # container's own filesystem does not.
    shift
    require_git_dir
    export_host_gids
    run_cli up --workspace-folder "$WORKSPACE" \
      --remove-existing-container "${DC_CONFIG[@]}" "$@"
    apply_cpu_limits
    ;;
  shell)
    shift
    # An INTERACTIVE shell must not route through run_cli, and the reason is
    # not the redaction but the pipe it needs to do it. Two things break once
    # the CLI's stdout is not a terminal:
    #
    #   1. sed withholds the prompt. A prompt ends in "$ ", not a newline, so
    #      a line-buffered filter holds it forever -- you get a blank screen
    #      and no echo of what you type, on a container that is working
    #      perfectly. (`sed -u` above fixes this much.)
    #   2. The terminal SIZE is lost regardless. The CLI decides whether to
    #      ask docker for a pty from process.stdin.isTTY, so a pty is still
    #      allocated, but it is sized from process.stdout.columns -- undefined
    #      through a pipe. Every full-screen program in there then draws into
    #      a default 80x24 that does not match your window.
    #
    # There is nothing to redact on this path anyway: GH_TOKEN reaches the
    # terminal from the `docker run` line that `up`/`rebuild` print on
    # failure, and `exec` prints no such line -- the token was baked into the
    # container's environment at create time.
    #
    # WITH arguments (`shell -c '...'`) it is a one-shot command rather than a
    # terminal session, so keep the filter: that is the path scripts and
    # agents use, and its output is what gets pasted into an issue.
    if [ $# -eq 0 ]; then
      exec "${CLI[@]}" exec --workspace-folder "$WORKSPACE" "${DC_CONFIG[@]}" bash
    fi
    run_cli exec --workspace-folder "$WORKSPACE" "${DC_CONFIG[@]}" bash "$@"
    ;;
  test)
    shift
    # TEST_CMD may carry its own arguments, so split it as a command line
    # rather than on whitespace alone.
    config_args "$TEST_CMD"
    cmd=("${CONFIG_ARGS[@]}")
    # -rs because tests that degrade-to-skip on a missing tool make a green run
    # meaningless until you have read the skip reasons.
    xdist_args
    cmd+=("${XDIST[@]}" -rs)
    # Default to the fast tier, unless the caller passed their own marker
    # expression -- then which tier runs is their call.
    case " $* " in
      *" -m "*) ;;
      *) config_args "$FAST_TEST_ARGS"; cmd+=("${CONFIG_ARGS[@]}") ;;
    esac
    run_cli exec --workspace-folder "$WORKSPACE" "${DC_CONFIG[@]}" \
      "${cmd[@]}" "$@"
    ;;
  down)
    shift
    # By default this takes down the ACTIVE variant only -- the one
    # DEVCONTAINER_CONFIG names -- because a cuda and a hip container for the
    # same worktree are two different containers that merely share a workspace
    # folder. `down --all` takes down every variant of this workspace, which
    # is what you want before `worktree.sh rm` or when you have lost track.
    scope=()
    [ "${1:-}" = --all ] && scope=(--any-config)
    ids=$(container_ids --stopped "${scope[@]}")
    if [ -z "$ids" ]; then
      echo "no container for $WORKSPACE${scope:+ (any config)}"
    else
      # One id per line, and there can be more than one, so loop: a single
      # quoted "$ids" would reach docker as one newline-joined name and fail
      # with "No such container" while leaving both behind.
      while read -r id; do
        [ -n "$id" ] || continue
        docker rm -f "$id"
      done <<<"$ids"
    fi
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "unknown command: $1" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac
