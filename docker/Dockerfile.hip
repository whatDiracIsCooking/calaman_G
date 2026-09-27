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
#   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.base -t gpumod:base .
#   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.hip  -t gpumod:hip .
#
# Both from the REPO ROOT: the trailing `.` is the context, and this file
# bind-mounts docker/install-rocm.sh from it.
#
# This is the image devtools/cross-backend-check.sh and `devtools/cpp-tier.sh
# --rocm` reach for (CROSS_CHECK_IMAGE / ROCM_IMAGE in devtools/config.sh), so
# it is worth having even on a machine with no AMD card -- the cross-backend
# check only compiles.
ARG PROJECT_NAME=gpumod
ARG PARENT_IMAGE=${PROJECT_NAME}:base
FROM ${PARENT_IMAGE}

ARG ROCM_VERSION=7.2.4

# Drop the ~12.9GB of ROCm that only a RUNNING kernel needs -- the Tensile and
# rocFFT kernel objects, composable-kernel's static archives, rccl, rocalution,
# hiptensor. See install-rocm.sh for the measured list and why nothing on it is
# a link-time dependency. OFF here: this is the dev image and it runs kernels.
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
ARG GPU_TARGETS=gfx1200
ENV GPU_TARGETS=${GPU_TARGETS} \
    AMDGPU_TARGETS=${GPU_TARGETS} \
    ROCM_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:${PATH}

USER ubuntu
RUN hipconfig --version
