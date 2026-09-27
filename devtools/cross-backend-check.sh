#!/usr/bin/env bash
# Compile the tree for the backend you are NOT building, to catch the class of
# bug that a green build on one backend cannot see.
#
#   devtools/cross-backend-check.sh [--fresh] [-j N] [--native] [--device-only]
#
# WHY THIS EXISTS. A build targets exactly one backend, and the two are
# compiled by different front ends: a .cu goes through nvcc under CUDA and
# through clang's `-x hip` under HIP. nvcc is the more permissive of the two,
# so code that only clang rejects compiles clean, passes ctest, and ships --
# and the first sign of trouble is a ROCm build weeks later.
#
# A concrete example of the class: a functor that carries a `const` member of
# CLASS type is non-trivially-copyable under clang, so it fails parallel_for's
# device_functor concept -- yet nvcc accepts it. Such code compiles clean, stays
# green across the whole CUDA tier, and only fails when someone builds ROCm. See
# parallel_for.cuh's device_functor comment for the rule.
#
# WHAT IT COSTS. Seconds. Measured cold with ccache disabled: 4s
# for --device-only, 7s for the whole compile-time tier (350 steps, dispatch
# checks included), and 7s is also the wall clock from the host with docker
# startup in it. That is the point rather than the exact digits: this is a
# COMPILE-ONLY check -- no GoogleTest fetch, no runtime tests, no device --
# which is why it costs seconds where devtools/cpp-tier.sh costs minutes. Do
# not let the cost note on cpp-tier.sh talk you out of running this one.
#
# WHAT IT DOES NOT DO. It proves the other backend COMPILES, not that it runs:
# no kernel is launched and no result is checked. A real ROCm box running
# `devtools/cpp-tier.sh --preset hip` is still the stronger statement, and is
# what you want before trusting the backend in anger. This is the cheap check
# you can afford every time.
#
# Flags:
#   --fresh         wipe the CMake cache first. Needed after changing a
#                   toolchain variable, which a plain reconfigure keeps.
#   -j N            parallel compile jobs (default: BUILD_JOBS from
#                   devtools/config.sh, or Ninja's own choice when empty).
#   --device-only   build only the device .cu libraries (~4s) instead of the
#                   whole compile-time tier (~7s). Narrower: it catches a
#                   kernel-side divergence but not one in a module unit, and
#                   it only knows the targets CROSS_CHECK_DEVICE_TARGETS names.
#   --native        skip the container and run cmake here. Implied when this
#                   host already has the target backend's toolchain.
#
# Exit status: 0 the other backend compiles, 1 it does not, 2 the check could
# not be run at all (no toolchain and no usable image) -- never a silent skip,
# because the machines least likely to have ROCm are exactly the ones where
# this is worth knowing.
#: -- help stops here --
#
# Deliberately NOT a git hook. It needs docker and a ~20GB image, so on a host
# without them it can only skip -- and a gate that silently does nothing on the
# machine that most needs it is worse than no gate, which is the same reasoning
# .pre-commit-config.yaml gives for keeping the C++ tier off the push gate.
# Run it before opening a PR that touches a .cu, or anything under src/ that a
# .cu includes.
#
# Deliberately NOT devcontainer.sh --hip either: that brings up a full
# devcontainer with its own volumes and per-worktree state, which is the right
# tool for working IN the HIP backend and far too much machinery for a
# four-second compile.
set -euo pipefail

# shellcheck source=./lib.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

usage() { usage_from_header "${BASH_SOURCE[0]}"; }

fresh=0
native=0
device_only=0
jobs=$BUILD_JOBS

while [ $# -gt 0 ]; do
  case "$1" in
    --fresh) fresh=1 ;;
    --native) native=1 ;;
    --device-only) device_only=1 ;;
    -j) shift; jobs=${1:?-j needs a value} ;;
    -j*) jobs=${1#-j} ;;
    -h|--help) usage; exit 0 ;;
    *) echo "cross-backend-check: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

backend=$CROSS_CHECK_BACKEND
preset=$CROSS_CHECK_PRESET
image=$CROSS_CHECK_IMAGE
build_dir=$CROSS_CHECK_BUILD_DIR
probe=$CROSS_CHECK_TOOL

# The targets that exist only to be compiled. Named rather than built wholesale
# so --device-only has something narrower to ask for; with neither, `cmake
# --build` builds everything the compile-time configuration generated.
declare -a device_targets=()
config_lines "$CROSS_CHECK_DEVICE_TARGETS"
device_targets=("${CONFIG_LINES[@]}")

# ---------------------------------------------------------------------------
# Phase 1: run the check, here, with cmake
# ---------------------------------------------------------------------------
run_here() {
  local -a cfg=(cmake --preset "$preset" -B "$build_dir" -DWWR_COMPILE_TIME_ONLY=ON)
  [ "$fresh" = 1 ] && cfg+=(--fresh)

  echo "cross-backend-check: configuring $backend (preset=$preset) -> $build_dir"
  "${cfg[@]}"

  local -a bld=(cmake --build "$build_dir")
  [ -n "$jobs" ] && bld+=(-j "$jobs")
  if [ "$device_only" = 1 ]; then
    bld+=(--target "${device_targets[@]}")
    echo "cross-backend-check: building ${#device_targets[@]} device librar(ies)"
  else
    echo "cross-backend-check: building the compile-time tier"
  fi
  "${bld[@]}"
}

# ---------------------------------------------------------------------------
# Phase 0: decide where to run it
# ---------------------------------------------------------------------------
# The toolchain for the OTHER backend is by definition not in the container you
# develop in -- the cuda image has no ROCm, and a HIP build needs no CUDA. So
# the normal path is a one-shot `docker run` against the other backend's image,
# from wherever you are, with no devcontainer involved.
if [ "$native" = 1 ] || command -v "$probe" >/dev/null 2>&1; then
  run_here
  echo "cross-backend-check: PASS ($backend compiles)"
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  cat >&2 <<EOF
cross-backend-check: cannot run -- no '$probe' on this machine and no docker.

  This needs either the $backend toolchain here, or the image that has it.
  devtools/doctor.sh reports which tools this machine actually has.
EOF
  exit 2
fi

if ! docker image inspect "$image" >/dev/null 2>&1; then
  cat >&2 <<EOF
cross-backend-check: cannot run -- docker image '$image' is not built.

  Build it once (it is large; build.sh walks the chain, since
  docker/Dockerfile.${backend,,} needs docker/Dockerfile.base tagged first):

    docker/build.sh ${backend,,}

  That tags <PROJECT_NAME>:${backend,,}, which is what CROSS_CHECK_IMAGE
  defaults to. Or set CROSS_CHECK_IMAGE to an image with the $backend toolchain.
EOF
  exit 2
fi

# --native inside, so the container does not re-derive any of this. The repo is
# bind-mounted at /workspace; .git is deliberately NOT mounted, since configure
# and build read none of it and a worktree's .git points outside the workspace.
# Running as the caller's uid keeps the build directory host-owned.
echo "cross-backend-check: no '$probe' here -- running in $image"
declare -a docker_args=(
  run --rm
  -u "$(id -u):$(id -g)"
  -v "$REPO_ROOT:/workspace"
  -w /workspace
  "$image"
  devtools/cross-backend-check.sh --native
)
[ "$fresh" = 1 ] && docker_args+=(--fresh)
[ "$device_only" = 1 ] && docker_args+=(--device-only)
[ -n "$jobs" ] && docker_args+=(-j "$jobs")

docker "${docker_args[@]}"
