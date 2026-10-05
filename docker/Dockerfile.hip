# syntax=docker/dockerfile:1.7
# ============================================================================
# hip -- base + ROCm, with no CUDA toolkit at all
# ============================================================================
# A HIP build needs no CUDA, which is the point of this being a sibling of
# Dockerfile.cuda rather than a layer on it: ROCm alone is ~30GB, and neither
# backend should pay for the other.
#
#   docker/build.sh hip       # builds base first, then this; tags :hip
#
# By hand, and the parent must already be tagged (see Dockerfile.base, "WHY THE
# FOUR FILES CHAIN BY TAG"):
#
#   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.base -t calaman:base .
#   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.hip  -t calaman:hip .
#
# Both from the REPO ROOT: the trailing `.` is the context, and this file
# bind-mounts docker/install-rocm.sh from it.
#
# This is the image devtools/cross-backend-check.sh and `devtools/cpp-tier.sh
# --rocm` reach for (CROSS_CHECK_IMAGE / ROCM_IMAGE in devtools/config.sh), so
# it is worth having even on a machine with no AMD card -- the cross-backend
# check only compiles.
ARG PROJECT_NAME=calaman
ARG PARENT_IMAGE=${PROJECT_NAME}:base
FROM ${PARENT_IMAGE}

ARG ROCM_VERSION=7.2.4

# Drop the ~12.1GB of ROCm that only a RUNNING kernel needs -- the Tensile and
# rocFFT kernel objects, composable-kernel's static archives, and rocalution.
# rccl and hiptensor are NOT dropped anymore: wwr.hip.rccl and wwr.hip.hiptensor
# link them, so pruning either would fail find_package at configure. See
# install-rocm.sh for the measured list and why nothing on it is a link-time
# dependency. OFF here: this is the dev image and it runs kernels.
# .github/workflows/images.yml sets it to 1 for the :hip-ci tag, whose only job
# is to answer "does the other backend still compile" on a GitHub-hosted runner
# that has neither an AMD card nor 20GB of free disk.
ARG ROCM_PRUNE=0
USER root

# Re-stamped rather than merely inherited from the parent, so that the
# PROJECT_NAME the devcontainer passes to THIS build is consumed where it is
# declared. See Dockerfile.base for what the label is for.
ARG PROJECT_NAME
LABEL devcontainer.project="${PROJECT_NAME}"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=bind,source=docker/install-rocm.sh,target=/tmp/install-rocm.sh \
    ROCM_VERSION="${ROCM_VERSION}" ROCM_PRUNE="${ROCM_PRUNE}" \
    bash /tmp/install-rocm.sh

# The LLVM target for the AMD card this image is built for. Exported as
# GPU_TARGETS / AMDGPU_TARGETS so a HIP CMake build has a default to read.
# gfx1200 is RDNA4 (Radeon RX 9060 XT); override for another card:
#   --build-arg GPU_TARGETS=gfx1100
#
# It is declared BEFORE the ROCm-DS layer below because that layer compiles
# kernels and has to be told what for -- and because hipCOMP takes ONE
# wave-size flag for the whole library, this single value is load-bearing there
# in a way it is not for the rest of the image. install-rocm-ds.sh has the
# detail, including why a multi-target list is rejected outright.
ARG GPU_TARGETS=gfx1200
ENV GPU_TARGETS=${GPU_TARGETS} \
    AMDGPU_TARGETS=${GPU_TARGETS} \
    ROCM_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:${PATH}

# hipCOMP: the one library this image needs that AMD does not package. Built
# from source into /opt/rocm-ds -- see install-rocm-ds.sh for why there is no
# apt name to use instead, and for why rocGRAPH and hipGRAPH are NOT here
# (they cannot be compiled for an RDNA target in usable time; issue #126).
#
# Dockerfile.combined declares this same arg and runs the same script, because
# it cannot inherit from this file. Bump them together, as with ROCM_VERSION
# and GPU_TARGETS.
ARG HIPCOMP_VERSION=v2.2.0
RUN --mount=type=bind,source=docker/install-rocm-ds.sh,target=/tmp/install-rocm-ds.sh \
    HIPCOMP_VERSION="${HIPCOMP_VERSION}" GPU_TARGETS="${GPU_TARGETS}" \
    bash /tmp/install-rocm-ds.sh

# /opt/rocm-ds is a second prefix beside /opt/rocm, so CMake has to be told
# about it; ROCm's own tree is already found through ROCM_PATH and hipcc.
ENV CMAKE_PREFIX_PATH=/opt/rocm-ds

# Claude Code, PINNED -- the last root layer in this file on purpose: a version
# bump re-runs only this and the two lines after it, never the SDK above it.
# docker/install-claude-code.sh carries the rest (why not Dockerfile.base, and
# which drift this ended). It is the fourth version to bump in more than one
# place: the same ARG is declared by Dockerfile.cuda, .hip and .combined, so
# bump it with `devtools/claude-version.sh --apply`, which rewrites all three
# at once -- doctor.sh warns when they have drifted apart.
#
# ENV as well as ARG so the INTENDED version is readable from inside a running
# container (`printenv CLAUDE_CODE_VERSION`), not merely inferable from the
# binary: that is what lets doctor.sh tell "pinned and installed" apart from
# "someone npm-installed over the top".
ARG CLAUDE_CODE_VERSION=2.1.289
ENV CLAUDE_CODE_VERSION=${CLAUDE_CODE_VERSION}
RUN --mount=type=bind,source=docker/install-claude-code.sh,target=/tmp/install-claude-code.sh \
    bash /tmp/install-claude-code.sh

USER ubuntu
RUN hipconfig --version
