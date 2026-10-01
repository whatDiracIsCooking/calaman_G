#!/usr/bin/env bash
# Install the ROCm/HIP SDK from AMD's apt repository.
#
# Called by docker/Dockerfile.hip, and again by docker/Dockerfile.combined on
# top of the CUDA image. A script rather than an inline RUN so the block is
# written once -- Dockerfiles have no include, and `combined` cannot inherit
# from two parents, so it reuses THIS instead of the hip image (see that file's
# header).
set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:?ROCM_VERSION must be set, e.g. 7.2.4}"

# AMD spells an X.Y.0 release's repo directory X.Y. Listing
# repo.radeon.com/rocm/apt/ shows 7.0, 7.0.1, 7.0.2, 7.0.3, 7.1, 7.2, 7.2.1 ...
# -- every .0 patch appears WITHOUT its trailing component, and there is no
# 7.0.0 directory at all. So the floor image (ROCM_VERSION=7.0.0, #111/#149)
# 404s on the Release file unless the .0 comes off here; the pin 7.2.4 is
# unaffected, which is why nothing caught this until the floor leg ran.
#
# ONLY the URL is rewritten. ROCM_VERSION stays the full MAJOR.MINOR.PATCH
# everywhere else -- build.sh's version-carrying tag (:hip-7.0.0-ci), doctor's
# comparison against /opt/rocm/.info/version, and the CMake floor all want the
# semver, and apt/7.0 does ship rocm-core 7.0.0.70000, so the two agree.
case "$ROCM_VERSION" in
  *.*.0) ROCM_APT_DIR="${ROCM_VERSION%.0}" ;;
  *)     ROCM_APT_DIR="$ROCM_VERSION" ;;
esac

# A signed-by keyring, not `apt-key add`: apt-key is deprecated in 24.04 and
# removed in 25.04.
#
# Only repo.radeon.com/rocm is added, NOT repo.radeon.com/amdgpu. That second
# repo exists to ship the DKMS kernel driver and AMD's libdrm fork, and neither
# belongs in a container -- the kernel driver comes from the host.
#
# The pin is load-bearing, not decoration. repo.radeon.com republishes its own
# builds of packages that also exist in Ubuntu; at equal priority apt is free to
# prefer either, and which one it picks can change between base-image refreshes.
# Priority 600 makes it deterministic -- ROCm's repo wins for anything it
# publishes. `o=repo.radeon.com` is the Origin its Release file declares.
install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
  | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
chmod 0644 /etc/apt/keyrings/rocm.gpg

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_APT_DIR} noble main" \
  > /etc/apt/sources.list.d/rocm.list
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
  > /etc/apt/preferences.d/rocm-pin-600

# rocm-hip-sdk is AMD's meta-package for "the HIP SDK" -- one name that tracks
# whatever AMD decides belongs in it (hipBLAS, hipSOLVER, hipFFT, hipRAND,
# hipSPARSE, rocm-smi, hipcc, rocm-cmake and the rest of the surface src/hip
# wraps).
#
# The five after it are the ones it does NOT pull, checked against its resolved
# dependency closure rather than assumed:
#   rocm-hip-runtime-dev  hipcc, rocm-llvm and lib/cmake/hip -- the HIP
#                   compiler itself. rocm-hip-sdk Depends on it at 7.0 and at
#                   7.2+, so naming it is a no-op there; the 7.1 SERIES DROPPED
#                   IT (verified in the apt index for both 7.1 and 7.1.1), and
#                   the omission surfaces only downstream, as CMake's
#                   "Failed to find ROCm root directory" when install-rocm-ds.sh
#                   calls enable_language(HIP). The floor image is 7.1, so this
#                   is load-bearing, not belt-and-braces -- and the tree needs
#                   hipcc regardless of what a meta-package decides to carry.
#   amd-smi-lib     the modern half of the nvml analogue (rocm-smi-lib, the
#                   older half, DOES come with the SDK via rocm-hip-libraries)
#   roctracer-dev   the cupti analogue
#   rocprofiler-sdk rocprofv3, its successor
#   rocm-gdb        parity with cuda-gdb
apt-get update
apt-get install -y --no-install-recommends \
  rocm-hip-sdk \
  rocm-hip-runtime-dev \
  amd-smi-lib \
  roctracer-dev \
  rocprofiler-sdk \
  rocm-gdb

# --- ROCM_PRUNE: the compile-only variant -----------------------------------
#
# ROCm installs at ~20GB. .github/workflows/images.yml builds a second tag
# (:hip-ci) with ROCM_PRUNE=1, which brings the image to 7.05GB; the dev image
# is untouched, since the default is off.
#
# IT IS AN OPTIMISATION, NOT AN ENABLER -- this comment used to claim a hosted
# runner has "20-25GB free" and that the prune was the only way the HIP job
# could fit. Measured on the first real run: a GitHub-hosted runner has a 145GB
# root with 86GB free BEFORE any cleanup, so the unpruned 20.5GB image would
# have fit comfortably. What the prune actually buys is ~13GB less to pull on
# every CI run, and a push that takes 3 minutes instead of many. Worth keeping
# for that, and worth not overstating.
#
# What comes out, measured rather than guessed (`du -x -d1 /opt/rocm/lib` plus
# `dpkg -S` on each of the largest files):
#
#   4.8G  composablekernel-dev   libdevice_{gemm,conv,reduction,contraction}
#                                _operations.a -- static archives nothing in
#                                this tree links. Pulled in by rocm-hip-sdk.
#   4.5G  hipblaslt/library      Tensile kernel objects, loaded at RUN time by
#   644M  rocblas/library        libhipblaslt.so / librocblas.so. The .so files
#   1.7G  rocfft/                themselves stay; only the kernel data goes.
#   459M  rocalution             sparse iterative solvers; nothing here wraps them
#
# ~12.1GB of ~20.5GB (it was ~12.3GB before hiptensor came off the list below),
# and none of it is a link-time dependency -- which is the
# test that decides what may go on this list. Adding anything here that a
# hipcc link actually needs surfaces as an undefined symbol in the ci-hip tier,
# not as a silent wrong answer, so the failure mode is at least loud.
#
# rccl (572M) used to be on this list -- "multi-GPU collectives; nothing here
# wraps them". wwr.hip.rccl now wraps it (issue #100), so roc::rccl is a
# link-time dependency of the ci-hip build and rccl/rccl-dev must STAY: pruning
# it would fail find_package(rccl) at configure, exactly the loud failure the
# rule above describes. The figures above predate its retention and are now
# ~570MB larger for it.
#
# hiptensor (226M) left this list the same way and for the same reason: it was
# "tensor contraction; nothing here wraps them" until the cuTENSOR/hipTensor
# pair was wrapped, and it is the ONE half of that pair that costs this image
# nothing extra to keep -- rocm-hip-sdk already installs hiptensor-dev through
# rocm-hip-libraries, where the CUDA side had to add a package (see
# docker/install-cuda.sh). Pruning it now fails find_package(hiptensor).
#
# composablekernel-dev STAYS on the list even though hiptensor is built on CK:
# what comes out is CK's static archives, which hiptensor consumed when IT was
# compiled. Linking libhiptensor.so does not re-link them.
#
# Since ci-hip became a full build it also LINKS and LOADS the runtime test
# binaries, so the SuiteListIsComplete guards now dlopen librocblas,
# librocsolver, librocsparse and librocfft out of a pruned tree on every CI
# run. They pass: those libraries read their kernel data lazily, on handle
# creation, not at load. That is a stronger check of this list than the
# compile-time tier could make.
#
# THE DELETION HAS TO HAPPEN IN THIS SCRIPT, not in a later Dockerfile layer.
# Layers are additive: an `rm` in a child layer hides the files but keeps their
# bytes in the parent, and the image does not shrink at all. That is also why
# :hip and :hip-ci cannot share the ROCm layer -- each is a full install.
#
# The whole packages go through apt rather than rm, so that removing one
# something else needs FAILS here instead of at link time. They pull out the
# rocm-hip-sdk / rocm-hip-libraries meta-packages with them, which carry no
# files of their own. --auto-remove is deliberately NOT passed: it would widen
# the removal to whatever else those metas were the last reference to, which is
# exactly the kind of quiet cascade this list is written to avoid. rccl/rccl-dev
# are deliberately NOT here anymore -- wwr.hip.rccl links roc::rccl (see the
# table above), and neither are hiptensor/hiptensor-dev.
if [ "${ROCM_PRUNE:-0}" = "1" ]; then
  echo "install-rocm.sh: ROCM_PRUNE=1 -- building the compile-only variant"
  apt-get purge -y \
    composablekernel-dev \
    rocalution rocalution-dev
  rm -rf \
    /opt/rocm/lib/hipblaslt/library \
    /opt/rocm/lib/rocblas/library \
    /opt/rocm/lib/rocfft
fi

# ROCm unpacks to /opt/rocm-<version> and rocm-core provides the /opt/rocm
# symlink. Nothing in the deb set registers that prefix with the dynamic loader,
# so without this every hipcc-linked binary would need an explicit RPATH.
echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf
ldconfig
test -L /opt/rocm

# Device access. /dev/kfd (the compute node) and /dev/dri/renderD* are
# root:render; /dev/dri/card* is root:video. The groups have to exist INSIDE the
# container for a non-root user to open those nodes once they are passed in, and
# `render` is not present in the stock Ubuntu image.
#
# gid 110 is Ubuntu 24.04's default for render, and the gid HERE is what a
# `--group-add render` by NAME would resolve to -- which is why nothing passes
# the name: the bind-mounted nodes keep the HOST's gid. Pass `--group-add
# <host-gid>` to docker run instead, which is what the devcontainer runArgs and
# cpp-tier.sh --rocm both do. Never rebuild the image to match a host.
(getent group render >/dev/null || groupadd -g 110 render || groupadd render)
(getent group video >/dev/null || groupadd video)
usermod -aG render,video ubuntu
