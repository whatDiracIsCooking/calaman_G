#!/usr/bin/env bash
# Prove the installed package works, by consuming it the way a stranger would.
#
#   devtools/install-check.sh [--preset NAME] [--prefix DIR] [--keep] [--no-run]
#
# Four steps, and the third is the only one that proves anything:
#
#   1. configure gpumod
#   2. cmake --install it into a throwaway prefix
#   3. configure example/consumer against ONLY that prefix, and build it
#   4. run the resulting binary
#
# Step 3 is the point. An install rule that runs without error still says
# nothing about whether the result is usable -- a missing module source, a
# header that did not travel next to the .cppm that includes it, a usage
# requirement that was PRIVATE and so never exported: all of those install
# perfectly cleanly and fail only in a consumer. example/consumer is a
# standalone project that knows nothing about this source tree and reaches
# gpumod through find_package alone, so building it is a real answer and
# `cmake --install` succeeding is not.
#
# WHY THIS EXISTS AS ITS OWN TIER. devtools/cpp-tier.sh builds and ctests the
# tree in place; nothing in it ever installs, so nothing in it can catch an
# export-set regression. A module added to src/ is picked up by the install
# sweep automatically (cmake/wwr_install.cmake reads the buildsystem back),
# but a module added with a PRIVATE compile requirement is exactly the change
# that passes cpp-tier.sh and breaks consumers. This is the tier that notices.
#
# THE TOOLCHAIN IS NOT ON YOUR HOST -- clang-20 with libc++'s module manifest
# and CMake 4.2 live in docker/Dockerfile.base, same as for cpp-tier.sh:
#
#   devtools/devcontainer.sh shell -c devtools/install-check.sh
#
# Step 4 needs a GPU; steps 1-3 do not need one to BUILD, but the `default`
# preset's CMAKE_CUDA_ARCHITECTURES=native queries a device at configure time,
# so on a GPU-less box pass --preset ci-cuda, which pins 86. The consumer
# binary exits 77 when it finds no device, which this script reports as a skip
# rather than a failure; --no-run stops before that and is what
# .github/workflows/ci.yml uses, since its runner has no card at all.
#
# Flags:
#   --preset NAME   gpumod configure preset (default: CMAKE_PRESET from
#                   devtools/config.sh).
#   --prefix DIR    install prefix (default: a mktemp -d, removed on exit).
#   --keep          keep the prefix and the consumer build dir, and print
#                   where they are. For inspecting what actually got installed.
#   --no-run        stop after building the consumer. Use on a box with no
#                   device when you only want the compile-and-link answer.
#
# Exit status is the first failing step's.
#: -- help stops here --
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

preset=$CMAKE_PRESET
prefix=""
keep=0
run=1

while [ $# -gt 0 ]; do
  case $1 in
    --preset) preset=${2:?--preset needs a value}; shift 2 ;;
    --prefix) prefix=${2:?--prefix needs a value}; shift 2 ;;
    --keep) keep=1; shift ;;
    --no-run) run=0; shift ;;
    -h | --help)
      sed -n '2,/^#: -- help stops here --$/p' "${BASH_SOURCE[0]}" |
        sed 's/^# \{0,1\}//; $d'
      exit 0
      ;;
    *) echo "install-check.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

# A build directory of its own. Sharing one with cpp-tier.sh would mean the two
# tiers reconfigure each other's cache back and forth -- a full rebuild each
# way -- which is the same reason every preset in CMakePresets.json has its own
# binaryDir.
build_dir=$REPO_ROOT/build-install-check
consumer_build=$REPO_ROOT/build-install-check-consumer

if [ -z "$prefix" ]; then
  prefix=$(mktemp -d -t gpumod-install-XXXXXX)
  created_prefix=1
else
  mkdir -p "$prefix"
  prefix=$(cd "$prefix" && pwd)
  created_prefix=0
fi

cleanup() {
  if [ "$keep" -eq 1 ]; then
    echo
    echo "kept: install prefix   $prefix"
    echo "kept: consumer build   $consumer_build"
    return
  fi
  [ "$created_prefix" -eq 1 ] && rm -rf "$prefix"
  rm -rf "$consumer_build"
}
trap cleanup EXIT

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
step "1/4  configure gpumod (preset: $preset)"
# ---------------------------------------------------------------------------
# -B overrides the preset's own binaryDir so this tier keeps its cache separate
# from cpp-tier.sh's, per the comment above.
cmake --preset "$preset" -B "$build_dir" -DWWR_INSTALL=ON

# ---------------------------------------------------------------------------
step "2/4  build and install into $prefix"
# ---------------------------------------------------------------------------
build_args=()
[ -n "${BUILD_JOBS:-}" ] && build_args+=(-j "$BUILD_JOBS")
cmake --build "$build_dir" "${build_args[@]}"
cmake --install "$build_dir" --prefix "$prefix"

echo
echo "installed:"
echo "  $(find "$prefix" -name '*.cppm' | wc -l) module interface sources"
echo "  $(find "$prefix" -name '*.a' | wc -l) static libraries"
echo "  $(find "$prefix" \( -name '*.h' -o -name '*.cuh' \) | wc -l) headers"

# The package config is what a consumer finds first; if it is missing, nothing
# downstream can work and the error there would not say so.
for required in \
  "lib/cmake/wwr/wwrConfig.cmake" \
  "lib/cmake/wwr/wwr-targets.cmake"; do
  if [ ! -f "$prefix/$required" ]; then
    echo "install-check.sh: FAIL -- $required was not installed" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
step "3/4  configure and build example/consumer against the install"
# ---------------------------------------------------------------------------
# CMAKE_PREFIX_PATH is the ONLY thing connecting the consumer to gpumod. No
# source path, no build directory, nothing from this tree -- if find_package
# cannot work from the install prefix alone, this step is where it shows.
rm -rf "$consumer_build"
cmake -S "$REPO_ROOT/example/consumer" -B "$consumer_build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$prefix"
cmake --build "$consumer_build" "${build_args[@]}"

# example/consumer builds one `consumer` target unconditionally, so a
# successful build (guaranteed by `set -e`) that leaves no binary is not a
# "disabled example" -- it is a real defect in the consumer build, and step 3
# is the only step that proves anything. Fail on it rather than passing.
if [ ! -x "$consumer_build/consumer" ]; then
  echo
  echo "install-check.sh: FAIL -- consumer build succeeded but produced no" >&2
  echo "  '$consumer_build/consumer' binary." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
if [ "$run" -eq 0 ]; then
  step "4/4  skipped (--no-run)"
  echo
  echo "install-check.sh: PASS (built, not run)"
  exit 0
fi

step "4/4  run the consumer"
# ---------------------------------------------------------------------------
set +e
"$consumer_build/consumer"
status=$?
set -e

if [ "$status" -eq 77 ]; then
  echo
  echo "install-check.sh: PASS (consumer built and linked; not run -- no GPU)"
  exit 0
fi
if [ "$status" -ne 0 ]; then
  echo
  echo "install-check.sh: FAIL -- consumer exited $status" >&2
  exit "$status"
fi

echo
echo "install-check.sh: PASS"
