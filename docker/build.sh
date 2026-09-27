#!/usr/bin/env bash
# Build one of docker/Dockerfile.{base,cuda,hip,combined}, and everything it
# inherits from, in order.
#
#   docker/build.sh base          <PROJECT_NAME>:base
#   docker/build.sh cuda          base, then cuda   -> :cuda and :latest
#   docker/build.sh hip           base, then hip    -> :hip
#   docker/build.sh combined      base, cuda, then combined -> :combined
#
# Anything after the target is passed through to EVERY `docker build` in the
# chain, which is what you want for --no-cache, --progress=plain or --pull:
#
#   docker/build.sh cuda --no-cache
#
# Run it from anywhere; the build context is always the repo root, because the
# Dockerfiles spell their COPY and bind-mount paths relative to it.
#
# Three environment knobs turn a local build into a published one, used only by
# .github/workflows/images.yml -- see "variant tags and the registry" below:
#
#   IMAGE_TAG_SUFFIX=-ci  IMAGE_REGISTRY=ghcr.io/owner  BUILD_PUSH=1 \
#     ROCM_PRUNE=1 docker/build.sh hip
#
# BUILD_DRY_RUN=1 prints every command, pushes included, and runs none of them.
#
# WHY THIS SCRIPT EXISTS
# The four Dockerfiles chain by TAG, not by stage (see Dockerfile.base, "WHY THE
# FOUR FILES CHAIN BY TAG"): a child's `FROM ${PARENT_IMAGE}` needs its parent
# already built and tagged, so something has to walk the chain in order. This is
# that something -- the single-Dockerfile `docker build --target cuda` that used
# to do it does not exist any more.
#
# It deliberately does NOT skip a step whose tag already exists. A cached
# `docker build` of an unchanged image is a second or two, and skipping on tag
# presence would silently hand a child a STALE parent after an edit to
# Dockerfile.base -- exactly the failure the single file could not have.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# PROJECT_NAME, and nothing else read here. Sourcing it is what keeps the image
# names in one place: devtools/config.sh already has to agree with the
# .devcontainer/*.json files (doctor.sh checks that), and a third spelling in
# here would be a third thing to drift.
# shellcheck source=../devtools/config.sh
. "$REPO_ROOT/devtools/config.sh"

export DOCKER_BUILDKIT=1

die() { echo "build.sh: $*" >&2; exit 1; }

TARGETS="base cuda hip combined"

target=${1:-}
[ -n "$target" ] || die "usage: docker/build.sh <${TARGETS// /|}> [docker build flags...]"
shift
# shellcheck disable=SC2076  # the literal spaces are the point: whole-word match
[[ " $TARGETS " == *" $target "* ]] || die "unknown target '$target' (want: $TARGETS)"

# --- the diamond, as three tables -------------------------------------------
# Kept as case statements rather than associative arrays so the file stays
# readable as the graph it describes.

parent_of() {
  case $1 in
    base)     echo "" ;;
    cuda|hip) echo "base" ;;
    combined) echo "cuda" ;;
  esac
}

# The tags each image gets. `cuda` gets :latest as well, because that is what
# docker/compose.yaml and README.md name (WWR_IMAGE defaults to it) -- the
# CUDA image is the default backend.
tags_of() {
  case $1 in
    base)     echo "$PROJECT_NAME:base" ;;
    cuda)     echo "$PROJECT_NAME:cuda $PROJECT_NAME:latest" ;;
    hip)      echo "$PROJECT_NAME:hip" ;;
    combined) echo "$PROJECT_NAME:combined" ;;
  esac
}

# Which build args each Dockerfile declares. Only names set in the environment
# are forwarded, so the DEFAULTS STAY IN THE DOCKERFILES and this script holds
# no version numbers of its own. Forwarding an arg to a file that does not
# declare it would earn a "build-args were not consumed" warning on every run,
# which is why these are per-file rather than one list.
build_args_of() {
  case $1 in
    base)         echo "UBUNTU_TAG LLVM_VERSION CMAKE_VERSION CMAKE_MAJOR_MINOR NINJA_VERSION" ;;
    cuda)         echo "CUDA_VERSION CUDA_ARCH" ;;
    hip)          echo "ROCM_VERSION GPU_TARGETS ROCM_PRUNE" ;;
    combined)     echo "ROCM_VERSION GPU_TARGETS" ;;
  esac
}

# --- variant tags and the registry ------------------------------------------
# Three environment knobs, none of which changes what is BUILT except through
# the build args above. .github/workflows/images.yml is the only caller that
# sets them; a local build sets none and behaves exactly as before.
#
#   IMAGE_TAG_SUFFIX=-ci   gpumod:hip        -> gpumod:hip-ci
#   IMAGE_REGISTRY=ghcr.io/owner             also tag ghcr.io/owner/gpumod:...
#   BUILD_PUSH=1           push the registry tags of the FINAL target
#
# The suffix exists so the pruned CI variant cannot overwrite the dev image
# that shares its Dockerfile -- same file, different build args, different tag.
# It is applied to every step of the chain, so a CI cuda is never built on a
# dev base someone has since rebuilt.
#
# `:latest` is DROPPED when a suffix is set. `gpumod:latest-ci` would be a
# lie -- latest is the alias docker/compose.yaml resolves by default, and it
# must keep meaning the full CUDA dev image.
#
# Only the FINAL target is pushed, never its parents: a child image is
# self-contained (FROM copies the layers in), so publishing :base as well would
# upload 1.45GB that nothing pulls.
IMAGE_TAG_SUFFIX=${IMAGE_TAG_SUFFIX:-}
IMAGE_REGISTRY=${IMAGE_REGISTRY:-}
BUILD_PUSH=${BUILD_PUSH:-}

if [ -n "$BUILD_PUSH" ] && [ -z "$IMAGE_REGISTRY" ]; then
  die "BUILD_PUSH is set but IMAGE_REGISTRY is not -- nothing to push to"
fi

# tags_of with IMAGE_TAG_SUFFIX applied. These are the LOCAL tags, and the
# first one is what a child takes as PARENT_IMAGE.
local_tags_of() {
  local tag
  for tag in $(tags_of "$1"); do
    if [ -z "$IMAGE_TAG_SUFFIX" ]; then
      echo "$tag"
    else
      case $tag in
        *:latest) ;;  # see above
        *) echo "${tag}${IMAGE_TAG_SUFFIX}" ;;
      esac
    fi
  done
}

registry_tags_of() {
  [ -n "$IMAGE_REGISTRY" ] || return 0
  local tag
  for tag in $(local_tags_of "$1"); do echo "${IMAGE_REGISTRY}/${tag}"; done
}

# --- resolve the chain, root first ------------------------------------------
chain=()
step=$target
while [ -n "$step" ]; do
  chain=("$step" "${chain[@]}")
  step=$(parent_of "$step")
done

echo "build.sh: $target <- ${chain[*]}  (project '$PROJECT_NAME', context $REPO_ROOT)"

for step in "${chain[@]}"; do
  read -r -a tags <<<"$(local_tags_of "$step" | tr '\n' ' ')"
  cmd=(docker build -f "$REPO_ROOT/docker/Dockerfile.$step")
  for tag in "${tags[@]}"; do cmd+=(-t "$tag"); done
  # Registry tags are applied at BUILD time, not by a later `docker tag`, so
  # the push below has nothing left to get wrong.
  if [ "$step" = "$target" ]; then
    while IFS= read -r tag; do
      # A here-string of the empty string still yields one empty line, which is
      # what this skips when IMAGE_REGISTRY is unset.
      [ -n "$tag" ] || continue
      cmd+=(-t "$tag")
    done <<<"$(registry_tags_of "$step")"
  fi

  # Always explicit, never left to the Dockerfile's default: a renamed project
  # must chain to ITS OWN parent, not to gpumod's.
  cmd+=(--build-arg "PROJECT_NAME=$PROJECT_NAME")
  parent=$(parent_of "$step")
  if [ -n "$parent" ]; then
    read -r -a parent_tags <<<"$(local_tags_of "$parent" | tr '\n' ' ')"
    cmd+=(--build-arg "PARENT_IMAGE=${parent_tags[0]}")
  fi

  for name in $(build_args_of "$step"); do
    # Set-but-empty still counts: it is how you ask for an empty value.
    [ -n "${!name+set}" ] && cmd+=(--build-arg "$name=${!name}")
  done

  cmd+=("$@" "$REPO_ROOT")

  echo "build.sh: + ${cmd[*]}"
  [ -n "${BUILD_DRY_RUN:-}" ] || "${cmd[@]}"
done

# Push last, and only the target's registry tags. Separate from the build loop
# so a failed push never leaves half a chain published, and so BUILD_DRY_RUN
# prints the push commands too.
if [ -n "$BUILD_PUSH" ]; then
  while IFS= read -r tag; do
    [ -n "$tag" ] || continue
    echo "build.sh: + docker push $tag"
    [ -n "${BUILD_DRY_RUN:-}" ] || docker push "$tag"
  done <<<"$(registry_tags_of "$target")"
fi

echo "build.sh: done -- $(for step in "${chain[@]}"; do local_tags_of "$step"; done | tr '\n' ' ')$(registry_tags_of "$target" | tr '\n' ' ')"
