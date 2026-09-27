#!/usr/bin/env bash
# The fast correctness gate, run at `git push` by pre-commit's pre-push stage.
# It is invoked ONLY when the push contains a .py change -- that condition is
# the `files: \.py$` filter on the hook in .pre-commit-config.yaml, not
# anything in here -- so by the time this runs there is Python to check.
#
# What it runs comes from devtools/config.sh: PREPUSH_PATHS (a curated subset,
# or everything when blank) filtered by FAST_TEST_ARGS. Keep that set to the
# tests that are seconds-fast and have no external dependency; anything that
# drives a compiler, a GPU or the network belongs in CI, not in a push gate --
# a gate people routinely bypass with --no-verify is not a gate.
#
# JOBS=n overrides the worker count for one run. Runnable by hand --
# `devtools/prepush-tests.sh` -- to see exactly what a push will run.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

PYTEST=$(find_pytest) || {
  echo "prepush-tests: no pytest found in ${VENV_PATHS} or on PATH" >&2
  echo "               run 'uv sync' (or set VENV_PATHS in devtools/config.sh)" >&2
  exit 1
}

xdist_args
config_args "$FAST_TEST_ARGS"; fast=("${CONFIG_ARGS[@]}")
config_args "$PREPUSH_PATHS";  paths=("${CONFIG_ARGS[@]}")

# -p no:cacheprovider keeps a push from writing .pytest_cache.
exec "$PYTEST" "${XDIST[@]}" -q -p no:cacheprovider \
  "${fast[@]}" "${paths[@]}" "$@"
