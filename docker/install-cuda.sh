#!/usr/bin/env bash
# Install the CUDA toolkit from NVIDIA's apt repository.
#
# Called by docker/Dockerfile.cuda, and inherited (already installed) by
# docker/Dockerfile.combined, which builds on the CUDA image. Still a script
# rather than an inline RUN, to match install-rocm.sh -- which has to be one,
# since `combined` runs it a second time.
#
# Why apt and not the nvidia/cuda base image: this repo builds a HIP backend
# too, and the shared docker/Dockerfile.base cannot be both nvidia/cuda and
# rocm/dev-ubuntu at once. Installing the toolkit on top of plain Ubuntu is what
# lets `base` stay vendor-neutral. The nvidia/cuda images are themselves Ubuntu
# plus these same packages, so nothing is lost -- except the NVIDIA_* runtime
# env those images set, which Dockerfile.cuda sets explicitly instead.
set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:?CUDA_VERSION must be set, e.g. 13-0}"
TARGETARCH="${TARGETARCH:-amd64}"

# NVIDIA publishes one repo directory per CPU architecture, under names that do
# not match Docker's TARGETARCH spelling.
case "$TARGETARCH" in
  amd64) repo_arch=x86_64 ;;
  arm64) repo_arch=sbsa ;;
  *) echo "unsupported TARGETARCH: '$TARGETARCH'" >&2; exit 1 ;;
esac

repo="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${repo_arch}"

# The keyring package registers both the signing key and the repo definition,
# which is why no sources.list line is written by hand here.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
wget -q -O "$tmp/cuda-keyring.deb" "${repo}/cuda-keyring_1.1-1_all.deb"
dpkg -i "$tmp/cuda-keyring.deb"

apt-get update
# The versioned meta-package, not `cuda` or `cuda-toolkit`: those pull the
# driver, which must come from the host through the NVIDIA container runtime.
# Installing a driver inside the image conflicts with the injected one.
#
# THE THREE AFTER IT ARE NOT IN THE TOOLKIT. Checked against the resolved
# closure of `cuda-toolkit-${CUDA_VERSION}`, which is compiler + libraries +
# tools + documentation + nvml and nothing else -- each of these ships as its
# own package from the SAME NVIDIA repo the keyring above registered, so they
# cost an apt name and no new source:
#
#   libnccl-dev       NCCL, multi-GPU collectives. wwr.cuda.nccl needs its
#                     nccl.h and libnccl.so -- the CUDA counterpart to ROCm's
#                     rccl, which IS in the base ROCm install. Unversioned so
#                     apt resolves the build matching the installed toolkit.
#                     See docker/install-rocm.sh for the HIP side.
#   cutensor-<major>  cuTENSOR, tensor contraction. Pairs with ROCm's hiptensor,
#                     which rocm-hip-sdk already installs. THE EXPENSIVE ONE:
#                     ~1.03GB installed (318MB runtime + 736MB dev at 2.8.1),
#                     and it lands in :cuda-ci, which ci.yml pulls on every run.
#   nvcomp-<major>    nvCOMP, GPU lossless compression. ~70MB. Its HIP
#                     counterpart is NOT packaged by AMD at all and is built
#                     from source; see docker/install-rocm-ds.sh.
#
# Why the `-cuda-<major>` suffix and not the bare `cutensor` / `nvcomp` metas:
# those are the CUDA-12-era names (`libcutensor-dev` still resolves to
# libcutensor2 2.2.0.0), while the suffixed metas track the toolkit generation
# installed here. The suffix is the MAJOR only -- `13`, not CUDA_VERSION's
# `13-0` spelling -- which is why it is derived rather than reused.
cuda_major="${CUDA_VERSION%%-*}"

apt-get install -y --no-install-recommends \
  "cuda-toolkit-${CUDA_VERSION}" \
  libnccl-dev \
  "cutensor-cuda-${cuda_major}" \
  "nvcomp-cuda-${cuda_major}"

# /usr/local/cuda is a symlink the packages maintain, so a 13.0 -> 13.1 bump
# does not leave anything pointing at a directory that no longer exists.
test -x /usr/local/cuda/bin/nvcc
/usr/local/cuda/bin/nvcc --version | grep release

# --- cuTENSOR and nvCOMP are NOT on any default search path ------------------
#
# libnccl-dev is a normal Debian library package: /usr/include/nccl.h and
# /usr/lib/<triplet>/libnccl.so, found by a bare `find_library`. The other two
# are not, and they are not even wrong in the same way -- each ships under its
# own package-named tree, so an unaided compile finds NEITHER header and the
# loader finds NEITHER runtime:
#
#   cuTENSOR   /usr/include/libcutensor/<major>/{cutensor.h,cutensor/types.h,…}
#              /usr/lib/<triplet>/libcutensor/<major>/libcutensor.so.2
#              and NO CMake config package at all -- find_path/find_library.
#   nvCOMP     /usr/include/libnvcomp<N>-dev-cuda-<major>/{nvcomp.h,nvcomp/…}
#              /usr/lib/<triplet>/libnvcomp<N>-dev-cuda-<major>/libnvcomp.so
#              -> ../libnvcomp<N>-cuda-<major>/libnvcomp.so.<N>  (the runtime)
#              and a real CMake config package, nvcomp-config.cmake.
#
# The `<N>` in nvCOMP's paths is the SONAME major, not the CUDA one, so those
# directory names change under a library bump that this file has no other
# reason to notice. Hence: DISCOVER the paths here, then publish them at fixed
# locations, rather than spelling today's names into Dockerfile.cuda's ENV
# where nothing would catch them going stale.
#
# The fixed shape is a /opt/nvidia/<lib>/{include,lib} prefix per library --
# what CMake already expects, so CMAKE_PREFIX_PATH is all Dockerfile.cuda has
# to set, and `find_package(nvcomp)` / `find_path(cutensor.h)` both resolve
# with no per-library variable. Symlinks, not copies: the bytes stay in the apt
# tree that owns them, and `apt remove` still empties this out.
triplet="$(basename "$(find /usr/lib -maxdepth 1 -type d -name '*-linux-gnu' | head -1)")"

cutensor_inc="/usr/include/libcutensor/${cuda_major}"
cutensor_lib="/usr/lib/${triplet}/libcutensor/${cuda_major}"
nvcomp_inc="$(find /usr/include -maxdepth 1 -type d -name "libnvcomp*-dev-cuda-${cuda_major}" | head -1)"
nvcomp_lib="$(find "/usr/lib/${triplet}" -maxdepth 1 -type d -name "libnvcomp*-dev-cuda-${cuda_major}" | head -1)"
nvcomp_cfg="$(dirname "$(find /usr/lib -name 'nvcomp-config.cmake' -print -quit)")"

for d in "$cutensor_inc" "$cutensor_lib" "$nvcomp_inc" "$nvcomp_lib" "$nvcomp_cfg"; do
  [ -d "$d" ] || { echo "install-cuda.sh: expected directory missing: '$d'" >&2; exit 1; }
done

install -d /opt/nvidia/cutensor /opt/nvidia/nvcomp/lib/cmake
ln -sfn "$cutensor_inc" /opt/nvidia/cutensor/include
ln -sfn "$cutensor_lib" /opt/nvidia/cutensor/lib
ln -sfn "$nvcomp_inc" /opt/nvidia/nvcomp/include
# nvcomp's own config dir is <cmake>/nvcomp, which is the name find_package
# looks for -- link the directory itself, not its parent.
ln -sfn "$nvcomp_cfg" /opt/nvidia/nvcomp/lib/cmake/nvcomp
for f in "$nvcomp_lib"/*; do ln -sfn "$f" "/opt/nvidia/nvcomp/lib/$(basename "$f")"; done

# Runtime. Neither directory is on the loader path, and cuTENSOR's .so.2 lives
# only in its versioned directory, so without this every binary linking either
# one would need an explicit RPATH. Same reasoning as ROCm's rocm.conf in
# install-rocm.sh. nvCOMP's dev symlinks point INTO the runtime package's
# directory, which is the one that has to be listed.
{
  echo "$cutensor_lib"
  find "/usr/lib/${triplet}" -maxdepth 1 -type d -name "libnvcomp*-cuda-${cuda_major}"
} > /etc/ld.so.conf.d/cuda-extras.conf
ldconfig

# Prove the whole path works, not just that files exist: ldconfig must resolve
# each SONAME by name, which is what a link against these will do.
for header in nccl.h cutensor.h nvcomp.h; do
  found="$(find /usr -name "$header" -print -quit)"
  [ -n "$found" ] || { echo "install-cuda.sh: $header not found after install" >&2; exit 1; }
  echo "install-cuda.sh: $header -> $found"
done
for soname in libnccl.so libcutensor.so libnvcomp.so; do
  ldconfig -p | grep -q "${soname}" \
    || { echo "install-cuda.sh: ${soname} not on the loader path" >&2; exit 1; }
done
echo "install-cuda.sh: cuTENSOR + nvCOMP published under /opt/nvidia"
