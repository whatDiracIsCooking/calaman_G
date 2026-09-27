#!/usr/bin/env bash
# Configure, build and ctest the C++ tree -- the tier that proves src/ still
# works, beside the pytest gate (which touches no built binary) and
# devtools/install-check.sh (which proves the INSTALLED package still works,
# something no amount of green here says anything about).
#
#   devtools/cpp-tier.sh [--preset NAME] [--fresh] [--clean] [--no-test]
#                        [--tidy] [--rocm] [-j N] [--report DIR]
#                        [-- <ctest args>]
#
# It is NOT on the push gate: a C++23
# named-module build with CUDA separable compilation costs minutes even warm,
# and a gate that costs minutes is one people learn to bypass with --no-verify.
# Run it before opening a PR, and let CI run it on a machine that has the
# toolchain.
#
# THE TOOLCHAIN IS NOT ON YOUR HOST. clang-20 with libc++'s module manifest,
# CMake 4.2 and nvcc live in docker/Dockerfile.cuda, so the normal way to call
# this is from inside that container (which DEVCONTAINER_CONFIG in config.sh
# already selects):
#
#   devtools/devcontainer.sh shell -c devtools/cpp-tier.sh
#
# On a bare host with its own clang and CUDA, use --preset workstation. The two
# presets are the same configuration and differ only in binaryDir -- nothing is
# expected prebuilt anywhere, since GoogleTest is fetched at configure time --
# so the point is simply that a host build and a container build get separate
# build directories instead of reconfiguring each other's.
# `devtools/doctor.sh` lists which of the two situations you are in.
#
# Flags:
#   --preset NAME   configure/build preset (default: CMAKE_PRESET from
#                   devtools/config.sh). The test preset follows it when one of
#                   the same name exists, and is skipped with a note when not
#                   -- only default, workstation, debug, asan, hip and
#                   compile-time have one.
#   --fresh         wipe the CMake cache first (`--fresh`). Needed after
#                   changing a toolchain variable; a plain reconfigure keeps
#                   the old value and the change appears to do nothing.
#   --clean         rebuild every object (`--clean-first`), keeping the cache.
#   --no-test       configure and build only.
#   --tidy          after a successful build, run clang-tidy over the .cpp
#                   implementation units using the build's
#                   compile_commands.json. Reports findings; does NOT fail
#                   the run, because the tree is not clean yet (see
#                   .clang-tidy). Off by default so the normal tier stays
#                   a build-and-test.
#   --rocm          run the HIP preset against a REAL AMD GPU, from the HOST.
#                   Re-execs this script inside ROCM_IMAGE with /dev/kfd,
#                   /dev/dri and the NUMERIC host render/video gids passed
#                   through (config.sh's ROCM_* settings say why numeric).
#                   Implies --preset hip unless --preset says otherwise.
#                   Every other flag is forwarded. Exits 2 rather than 0 when
#                   it cannot run, so a skip is never read as a pass.
#                   cross-backend-check.sh proves the other backend COMPILES;
#                   this proves it RUNS, and needs a card to say so.
#   -j N            parallel compile jobs (default: BUILD_JOBS from config.sh,
#                   or Ninja's own choice when that is empty). Module builds
#                   are memory-hungry per job -- a too-high N OOMs rather than
#                   failing cleanly.
#   --report DIR    timestamped log + one-line summary (default:
#                   SLOW_TIER_REPORTS from config.sh).
#   -- <args>       everything after `--` is passed to ctest verbatim
#                   (e.g. `-- -R Constants` to scope the run). ctest has
#                   one entry per gtest case, so -R matches suite names.
#
# Exit status is the first failing phase's, so a scheduler can alert on it.
#: -- help stops here --
#
# Deliberately NOT what docker/compose.yaml's `build`/`test` services do, even
# though the commands overlap: compose owns the batch path (it brings its own
# container, tees to .log/, and is the thing CI calls), this owns the
# already-inside-a-container path. Changing the preset list means touching
# CMakePresets.json, which both read -- neither hardcodes one.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

preset=$CMAKE_PRESET
test_preset=$CTEST_PRESET
fresh=0
clean=0
run_tests=1
tidy=0
rocm=0
jobs=$BUILD_JOBS
report_dir=$SLOW_TIER_REPORTS
declare -a passthrough=()
preset_explicit=0

while [ $# -gt 0 ]; do
  case "$1" in
    --preset)  preset=$2; preset_explicit=1; shift 2;;
    --fresh)   fresh=1; shift;;
    --clean)   clean=1; shift;;
    --no-test) run_tests=0; shift;;
    --tidy)    tidy=1; shift;;
    --rocm)    rocm=1; shift;;
    -j|--jobs) jobs=$2; shift 2;;
    --report)  report_dir=$2; shift 2;;
    --)        shift; passthrough=("$@"); break;;
    -h|--help) usage_from_header "${BASH_SOURCE[0]}"; exit 0;;
    *)         echo "cpp-tier: unknown flag: $1" >&2; exit 2;;
  esac
done

# --preset switches the test preset with it, or the run silently tests the
# wrong build directory: ctest --preset default reads build/, while
# --preset asan built into build-asan/.
[ "$preset_explicit" -eq 1 ] && test_preset=$preset

# --rocm: re-exec inside the ROCm image with the GPU attached, then stop.
#
# This is the one path where cpp-tier.sh launches a container rather than
# assuming it is already in one, and it is here rather than in a script of its
# own because everything after this point -- preset handling, the test-preset
# fallback, logging, --tidy -- is what we want to run, unchanged, on the other
# side. The container calls this same script without --rocm.
if [ "$rocm" = 1 ]; then
  [ "$preset_explicit" -eq 1 ] || { preset=$ROCM_PRESET; test_preset=$ROCM_PRESET; }

  command -v docker >/dev/null 2>&1 || {
    echo "cpp-tier --rocm: cannot run -- no docker on this machine." >&2
    exit 2
  }

  docker image inspect "$ROCM_IMAGE" >/dev/null 2>&1 || {
    cat >&2 <<EOF
cpp-tier --rocm: cannot run -- docker image '$ROCM_IMAGE' is not built.

  Build it once (build.sh walks the chain -- docker/Dockerfile.hip needs
  docker/Dockerfile.base built and tagged first):

    docker/build.sh hip

  That tags <PROJECT_NAME>:hip, which is what ROCM_IMAGE defaults to. Or set
  ROCM_IMAGE to an image that has the ROCm toolchain.
EOF
    exit 2
  }

  # The device nodes, and the gids that make them openable. A missing /dev/kfd
  # means no amdgpu -- worth naming, because the alternative is a build that
  # succeeds and then reports no device, which reads like a code problem.
  declare -a dev_args=()
  while IFS= read -r dev; do
    [ -n "$dev" ] || continue
    [ -e "$dev" ] || {
      echo "cpp-tier --rocm: cannot run -- $dev does not exist." >&2
      echo "  No AMD compute device here. Is amdgpu loaded? (lsmod | grep amdgpu)" >&2
      exit 2
    }
    dev_args+=(--device="$dev")
  done <<<"$ROCM_DEVICES"

  # NUMERIC, resolved on the HOST. See config.sh's ROCM_GROUPS: passing the
  # NAME lets the container resolve it against its own /etc/group, and the run
  # then fails exactly like a machine with no GPU.
  declare -a group_args=()
  while IFS= read -r grp; do
    [ -n "$grp" ] || continue
    # Numeric HOST gid, or empty; host_gid (lib.sh) owns the getent exit-2 trap.
    gid=$(host_gid "$grp")
    [ -n "$gid" ] || {
      echo "cpp-tier --rocm: cannot run -- no '$grp' group on this host." >&2
      echo "  getent group $grp found nothing; ROCM_GROUPS in config.sh names it." >&2
      exit 2
    }
    group_args+=(--group-add "$gid")
  done <<<"$ROCM_GROUPS"

  # Mounted at its own host path, not /workspace, for two reasons: a CMake
  # cache records absolute directories, so this shares build-hip/ with a
  # devcontainer build instead of fighting it; and `git rev-parse` keeps
  # working, which is what puts the revision in the log name. A worktree's
  # .git points into the main checkout, outside this tree, so that directory
  # is mounted too when it is not already under us.
  declare -a mount_args=(-v "$REPO_ROOT:$REPO_ROOT")
  git_common=$(git -C "$REPO_ROOT" rev-parse --git-common-dir 2>/dev/null || true)
  case "$git_common" in
    "")     ;;
    /*)     ;;
    *)      git_common="$REPO_ROOT/$git_common";;
  esac
  if [ -n "$git_common" ] && [ -d "$git_common" ]; then
    git_common=$(cd "$git_common" && pwd)
    case "$git_common" in
      "$REPO_ROOT"/*) ;;
      *) mount_args+=(-v "$git_common:$git_common");;
    esac
  fi

  # Rebuilt from parsed state rather than replayed from "$@", so --rocm is
  # dropped and the resolved preset travels explicitly.
  declare -a inner=(devtools/cpp-tier.sh --preset "$preset" --report "$report_dir")
  [ "$fresh" = 1 ]     && inner+=(--fresh)
  [ "$clean" = 1 ]     && inner+=(--clean)
  [ "$run_tests" = 0 ] && inner+=(--no-test)
  [ "$tidy" = 1 ]      && inner+=(--tidy)
  [ -n "$jobs" ]       && inner+=(-j "$jobs")
  [ ${#passthrough[@]} -gt 0 ] && inner+=(-- "${passthrough[@]}")

  echo "cpp-tier --rocm: $ROCM_IMAGE, preset=$preset, devices=$(echo $ROCM_DEVICES)"
  exec docker run --rm \
    "${dev_args[@]}" "${group_args[@]}" \
    -u "$(id -u):$(id -g)" -e HOME=/tmp \
    "${mount_args[@]}" -w "$REPO_ROOT" \
    "$ROCM_IMAGE" "${inner[@]}"
fi

for tool in cmake ninja; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "cpp-tier: $tool not found -- this needs the CUDA container:" >&2
    echo "  devtools/devcontainer.sh shell -c devtools/cpp-tier.sh" >&2
    echo "(or --preset workstation, on a host with its own clang and CUDA)" >&2
    exit 1
  }
done

case "$report_dir" in /*) ;; *) report_dir="$REPO_ROOT/$report_dir";; esac
mkdir -p "$report_dir"
stamp=$(date +%Y%m%d-%H%M%S)
rev=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
log="$report_dir/cpp-$stamp-$rev-$preset.log"

echo "cpp-tier: $REPO_ROOT @ $rev  preset=$preset -> $log"

# Where this preset puts compile_commands.json. Asked of CMake rather than
# guessed: `default` uses build/ but the others use build-<preset>/, and a
# hardcoded build/ would have --tidy silently lint the previous preset's
# database.
build_dir=$(cmake --preset "$preset" -N 2>/dev/null |
  sed -n 's/^.*Binary directory: *//p' | head -1)
[ -n "$build_dir" ] || build_dir="$REPO_ROOT/build"

# `jq -e` is not assumed to be present, so the "does this test preset exist"
# question is answered by asking ctest, which is. --show-only lists the tests a
# preset would run and exits non-zero when the preset is unknown.
has_test_preset() {
  [ -n "$test_preset" ] || return 1
  ctest --preset "$test_preset" --show-only >/dev/null 2>&1
}

run_phase() {
  local name=$1; shift
  echo "--- $name: $* ---" | tee -a "$log"
  "$@" 2>&1 | tee -a "$log"
  return "${PIPESTATUS[0]}"
}

rc=0
set +e

configure=(cmake --preset "$preset")
[ "$fresh" -eq 1 ] && configure+=(--fresh)
run_phase configure "${configure[@]}" || rc=$?

if [ "$rc" -eq 0 ]; then
  build=(cmake --build --preset "$preset")
  [ -n "$jobs" ] && build+=(-j "$jobs")
  [ "$clean" -eq 1 ] && build+=(--clean-first)
  run_phase build "${build[@]}" || rc=$?
fi

if [ "$rc" -eq 0 ] && [ "$run_tests" -eq 1 ]; then
  if has_test_preset; then
    run_phase ctest ctest --preset "$test_preset" "${passthrough[@]}" || rc=$?
  else
    echo "cpp-tier: no test preset '${test_preset:-<empty>}' -- built only" \
      | tee -a "$log"
    # Ask CMakePresets.json rather than naming the set here: a hardcoded list
    # goes stale the first time a preset is added, and this message is the one
    # place someone looks to find out which presets can actually run ctest.
    # `ctest --list-presets` prints a header and then two-space-indented,
    # quoted names.
    avail=$(ctest --list-presets 2>/dev/null |
      sed -n 's/^  "\([^"]*\)".*/\1/p' | paste -sd, - | sed 's/,/, /g')
    echo "          (test presets in CMakePresets.json: ${avail:-none found})" \
      | tee -a "$log"
  fi
fi

# clang-tidy is deliberately advisory -- run on request (--tidy), never folded
# into rc. It lints the src/*.cpp implementation units; the .cppm interfaces they
# import are governed by .clang-tidy's Header/ExcludeHeaderFilterRegex, which
# drops the vendor re-export layer (the same set config.sh hides from coverage,
# COVERAGE_IGNORE_REGEX) and, together with src/wrappers/.clang-tidy and the
# gpu*-prefix exemptions, silences the names and signatures that mirror the
# vendor API by design. With those in place the src/*.cpp run is clean today, so
# folding it into rc is now a small step -- left advisory only because
# clang-tidy's C++23-module support is still incomplete and can emit the odd
# false diagnostic on a module construct, which a build gate should not fail on.
if [ "$rc" -eq 0 ] && [ "$tidy" -eq 1 ]; then
  if ! command -v clang-tidy >/dev/null 2>&1; then
    echo "cpp-tier: clang-tidy not found -- skipping --tidy" | tee -a "$log"
  else
    mapfile -t tidy_files < <(git -C "$REPO_ROOT" ls-files 'src/*.cpp')
    if [ "${#tidy_files[@]}" -eq 0 ]; then
      echo "cpp-tier: no src/*.cpp to tidy" | tee -a "$log"
    else
      # Advisory: the phase's own status is reported but never folded into rc.
      run_phase clang-tidy clang-tidy -p "$build_dir" --quiet "${tidy_files[@]}" \
        || echo "cpp-tier: clang-tidy reported findings (advisory)" | tee -a "$log"
    fi
  fi
fi

set -e

verdict=$([ "$rc" -eq 0 ] && echo PASS || echo "FAIL(rc=$rc)")
summary="cpp-tier $verdict  rev=$rev  preset=$preset  stamp=$stamp"
echo "$summary" | tee -a "$log"
echo "$summary" >> "$report_dir/summary.log"
exit "$rc"
