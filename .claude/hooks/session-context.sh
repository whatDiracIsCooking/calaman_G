#!/usr/bin/env bash
# SessionStart hook: print one line telling the session which SIDE it is on
# (bare host vs inside the devcontainer) and which CHECKOUT it is in. The two
# are orthogonal -- you can be host+worktree, container+main, etc. -- and only
# the side actually changes what you should do (host vs container pytest worker
# counts, .venv vs /opt/venv, whether cmake exists at all). The harness startup
# block already states the checkout; this adds the side, which it does not.
#
# SessionStart stdout is appended to the session context, so keep it to one line.
set -euo pipefail

# --- side: host vs container -------------------------------------------------
# /.dockerenv is the canonical container marker; /opt/venv is this repo's own
# in-image venv (VENV_PATHS in devtools/config.sh lists it as the container one).
if [ -f /.dockerenv ] || [ -d /opt/venv ]; then
  side="container"
else
  side="host"
fi

# --- GPU presence ------------------------------------------------------------
# Report which GPU vendors are actually USABLE from here -- the thing that
# decides whether `native` configures and device work runs. It matters on BOTH
# sides: the host runs `cpp-tier.sh --rocm` and `native` CUDA configure too, not
# just the container. Each probe queries the driver, not just the device nodes,
# on purpose: devcontainer.sh's CPU-limit path runs `docker update`, which
# revokes the NVIDIA device cgroup while leaving every /dev/nvidia* node in
# place, and nvidia-smi then fails NVML -- exactly the "no GPU" we want to
# report. For AMD, amdgpu-arch is not on PATH in this image, so rocminfo is the
# probe; gate it on /dev/kfd so a box with no AMD stack pays nothing for the
# slower call.
#
# grep -c (not -q) on purpose: -q exits on the first match and closes the pipe,
# killing the producer with SIGPIPE, and under `set -o pipefail` that non-zero
# propagates and the `if` reads a present GPU as absent -- a timing race that
# bit rocminfo (many lines after the match) and would bite nvidia-smi on a
# multi-GPU host. -c reads the whole stream, so no early close.
gpus=()
if command -v nvidia-smi >/dev/null 2>&1 \
   && nvidia-smi -L 2>/dev/null | grep -c '^GPU ' >/dev/null; then
  gpus+=("NVIDIA")
fi
if [ -e /dev/kfd ] && command -v rocminfo >/dev/null 2>&1 \
   && rocminfo 2>/dev/null | grep -c 'Device Type:[[:space:]]*GPU' >/dev/null; then
  gpus+=("AMD")
fi
if [ "${#gpus[@]}" -gt 0 ]; then
  joined=$(IFS=,; printf '%s' "${gpus[*]}")
  side="$side (GPU: ${joined//,/, })"
else
  side="$side (no GPU)"
fi

# --- checkout: main vs worktree ----------------------------------------------
# In a linked worktree the per-worktree git dir differs from the common one.
checkout="main-checkout"
common=$(git rev-parse --git-common-dir 2>/dev/null || true)
gitdir=$(git rev-parse --git-dir 2>/dev/null || true)
if [ -n "$common" ] && [ "$common" != "$gitdir" ]; then
  top=$(git rev-parse --show-toplevel 2>/dev/null || true)
  case "$top" in
    */.claude/worktrees/*) checkout="worktree (lightweight, .claude/worktrees/)" ;;
    *)                      checkout="worktree (container-backed sibling)" ;;
  esac
fi

echo "Environment: $side · $checkout"
