#!/usr/bin/env bash
# Collect clang source-based coverage for the C++ runtime tier and print an
# llvm-cov summary for src/.
#
#   devtools/coverage.sh [--preset NAME] [--fresh] [--html] [-j N]
#                        [-- <ctest args>]
#
# This is the coverage counterpart to devtools/cpp-tier.sh: it configures and
# builds the tree with instrumentation (the `coverage` preset sets
# WWR_COVERAGE=ON, which adds -fprofile-instr-generate -fcoverage-mapping to
# CXX only -- nvcc's .cu units are untouched, exactly as ASAN is), runs ctest so
# the instrumented binaries drop .profraw files, then merges them with
# llvm-profdata and reports with llvm-cov.
#
# WHAT THIS ACTUALLY MEASURES. A large part of src/ is C++23 modules whose
# correctness is proved at COMPILE time (static_asserts, the dispatch checks),
# and those lines never execute -- so counting them would only drag the number
# down with lines already tested a different way. The two purely compile-time
# groups (the vendor re-exports src/cuda|src/hip, and the dispatch-checked
# neutral wrappers src/{blas,complex,fft,rand,solver,sparse}.cppm) are therefore
# dropped from the summary via COVERAGE_IGNORE_REGEX in config.sh. What remains
# is code the runtime suites are meant to reach. Read the number as "which of the
# runtime-relevant src/ lines the executing tests touch" -- still a floor, not a
# grade, since a header template line only shows covered once some suite
# instantiates and runs it.
#
# THE TOOLCHAIN IS NOT ON YOUR HOST, same as cpp-tier.sh: clang-20 with libc++
# and the matching llvm-cov/llvm-profdata live in the container, so the normal
# way to call this is from inside it:
#
#   devtools/devcontainer.sh shell -c devtools/coverage.sh
#
# The llvm tools MUST match the clang that built the tree -- a .profraw carries a
# format version, and a mismatched llvm-profdata rejects it. This resolves the
# versioned names (llvm-cov-20) first for that reason.
#
# Flags:
#   --preset NAME   configure/build/test preset (default: COVERAGE_PRESET from
#                   devtools/config.sh -- `coverage`). Any preset works, but it
#                   must have WWR_COVERAGE=ON or there is nothing to collect.
#   --fresh         wipe the CMake cache first (`--fresh`).
#   --html          also write a browsable HTML report under the build dir
#                   (build-coverage/coverage/html/index.html).
#   -j N            parallel compile jobs (default: BUILD_JOBS from config.sh).
#   -- <args>       passed to ctest verbatim (e.g. `-- -R Conversion` to scope
#                   which suites run, and thus what gets covered).
#
# Artifacts land under the preset's build dir -- build-coverage/coverage/ --
# which the /build*/ line in .gitignore already ignores, so there is nothing to
# clean up by hand. Exit status is the first failing phase's.
#: -- help stops here --
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

preset=$COVERAGE_PRESET
fresh=0
html=0
jobs=$BUILD_JOBS
declare -a passthrough=()

while [ $# -gt 0 ]; do
  case "$1" in
    --preset)  preset=$2; shift 2;;
    --fresh)   fresh=1; shift;;
    --html)    html=1; shift;;
    -j|--jobs) jobs=$2; shift 2;;
    --)        shift; passthrough=("$@"); break;;
    -h|--help) usage_from_header "${BASH_SOURCE[0]}"; exit 0;;
    *)         echo "coverage: unknown flag: $1" >&2; exit 2;;
  esac
done

for tool in cmake ninja; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "coverage: $tool not found -- this needs the container:" >&2
    echo "  devtools/devcontainer.sh shell -c devtools/coverage.sh" >&2
    exit 1
  }
done

# The llvm pair, version-matched to the clang that built the tree. The bare
# names are off PATH in this image (they live in /usr/lib/llvm-20/bin), so try
# the versioned name and that directory before giving up.
pick_llvm() {
  local base=$1 n
  for n in "${base}-20" "$base"; do
    command -v "$n" >/dev/null 2>&1 && { echo "$n"; return 0; }
  done
  for n in /usr/lib/llvm-*/bin/"$base"; do
    [ -x "$n" ] && { echo "$n"; return 0; }
  done
  return 1
}
llvm_cov=$(pick_llvm llvm-cov) || {
  echo "coverage: llvm-cov not found (need the one matching clang-20)." >&2
  exit 1
}
llvm_profdata=$(pick_llvm llvm-profdata) || {
  echo "coverage: llvm-profdata not found (need the one matching clang-20)." >&2
  exit 1
}

# Where this preset builds. Asked of CMake rather than guessed -- the coverage
# preset uses build-coverage/, but --preset could name another.
build_dir=$(cmake --preset "$preset" -N 2>/dev/null |
  sed -n 's/^.*Binary directory: *//p' | head -1)
[ -n "$build_dir" ] || build_dir="$REPO_ROOT/build-coverage"

cov_dir="$build_dir/coverage"
raw_dir="$cov_dir/raw"
profdata="$cov_dir/coverage.profdata"
report_txt="$cov_dir/report.txt"

echo "coverage: preset=$preset  build=$build_dir"
echo "coverage: llvm-cov=$llvm_cov  llvm-profdata=$llvm_profdata"

# --- configure + build ----------------------------------------------------
configure=(cmake --preset "$preset")
[ "$fresh" -eq 1 ] && configure+=(--fresh)
echo "--- configure: ${configure[*]} ---"
"${configure[@]}"

build=(cmake --build --preset "$preset")
[ -n "$jobs" ] && build+=(-j "$jobs")
echo "--- build: ${build[*]} ---"
"${build[@]}"

# --- run the instrumented tests -------------------------------------------
# One .profraw per process: %p (pid) keeps the two conversion suites, which run
# as separate ctest entries, from clobbering each other; %m adds the binary
# signature so distinct binaries never collide either. llvm-profdata merges the
# lot. Wipe the previous run's raws first so a removed test cannot leave stale
# counts behind.
rm -rf "$raw_dir"
mkdir -p "$raw_dir"
echo "--- ctest: --preset $preset ${passthrough[*]} ---"
LLVM_PROFILE_FILE="$raw_dir/cov-%p-%m.profraw" \
  ctest --preset "$preset" "${passthrough[@]}"

shopt -s nullglob
raws=("$raw_dir"/*.profraw)
shopt -u nullglob
[ "${#raws[@]}" -gt 0 ] || {
  echo "coverage: no .profraw produced -- did any instrumented binary run?" >&2
  echo "  (a preset without WWR_COVERAGE=ON collects nothing.)" >&2
  exit 1
}
echo "coverage: ${#raws[@]} .profraw file(s)"

# --- merge ----------------------------------------------------------------
"$llvm_profdata" merge -sparse "${raws[@]}" -o "$profdata"

# --- enumerate the instrumented binaries ctest ran ------------------------
# llvm-cov needs the binaries by hand; ask ctest which ones it ran rather than
# hardcoding a list that goes stale (the repo's standing complaint about
# hand-maintained lists). Keep only argv[0]s that live UNDER the build dir and
# are executable -- that drops the dispatch checks' llvm-objdump/python and the
# suite-guard's cmake, which are real tests but not instrumented objects.
mapfile -t objects < <(
  ctest --test-dir "$build_dir" --show-only=json-v1 2>/dev/null |
    python3 -c '
import json, os, sys
build = os.path.realpath(sys.argv[1])
seen, out = set(), []
for t in json.load(sys.stdin).get("tests", []):
    cmd = t.get("command") or []
    if not cmd:
        continue
    exe = os.path.realpath(cmd[0])
    if not exe.startswith(build + os.sep):
        continue
    if not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
        continue
    if exe in seen:
        continue
    seen.add(exe)
    out.append(exe)
print("\n".join(out))
' "$build_dir"
)
[ "${#objects[@]}" -gt 0 ] || {
  echo "coverage: found no instrumented test binaries under $build_dir." >&2
  exit 1
}
echo "coverage: ${#objects[@]} instrumented binary/binaries covered"

# llvm-cov takes the first object positionally and the rest via -object.
declare -a obj_args=("${objects[0]}")
for o in "${objects[@]:1}"; do obj_args+=(-object "$o"); done

# --- report ---------------------------------------------------------------
# Restrict to src/ (the trailing path filter) -- the report is about the
# project's own code, not GoogleTest or the test files themselves. Then drop the
# compile-time-only files (COVERAGE_IGNORE_REGEX from config.sh: the vendor
# re-exports and the dispatch-checked neutral wrappers) so the number reflects
# what the runtime suites actually reach -- see that variable's comment.
declare -a ignore_args=()
[ -n "${COVERAGE_IGNORE_REGEX:-}" ] && ignore_args=(-ignore-filename-regex="$COVERAGE_IGNORE_REGEX")

mkdir -p "$cov_dir"
"$llvm_cov" report "${obj_args[@]}" \
  -instr-profile="$profdata" \
  "${ignore_args[@]}" \
  "$REPO_ROOT/src" | tee "$report_txt"
echo "coverage: text report -> $report_txt"

if [ "$html" -eq 1 ]; then
  html_dir="$cov_dir/html"
  "$llvm_cov" show "${obj_args[@]}" \
    -instr-profile="$profdata" \
    "${ignore_args[@]}" \
    -format=html -output-dir="$html_dir" \
    "$REPO_ROOT/src"
  echo "coverage: HTML report -> $html_dir/index.html"
fi
