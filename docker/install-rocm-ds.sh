#!/usr/bin/env bash
# Build hipCOMP from source into /opt/rocm-ds.
#
# Called by docker/Dockerfile.hip AND by docker/Dockerfile.combined. Combined
# takes the CUDA image as its parent and re-runs the HIP side's scripts rather
# than inheriting them (see that file's header), so this is the second script
# after install-rocm.sh that BOTH files drive and whose pin therefore has to be
# bumped in both places.
#
# WHY SOURCE, WHEN EVERY OTHER ROCm LIBRARY HERE IS ONE APT NAME
# Because AMD publishes no package for it. Checked against the ROCm 7.2.4 apt
# index rather than assumed: its 415 packages include hiptensor, rccl, migraphx
# and composablekernel, and contain no hipcomp. ROCm-DS -- AMD's RAPIDS
# counterpart, where hipCOMP lives -- has no apt channel at all:
# repo.radeon.com/rocm-ds exists and is empty of packages. So the choice is not
# "source vs apt", it is "source or not at all".
#
# WHY THE PREFIX AND THIS FILE ARE STILL NAMED FOR ROCm-DS RATHER THAN hipCOMP
# Because hipCOMP is one of several ROCm-DS libraries and the others are
# expected back. rocGRAPH and hipGRAPH were built here too, and were removed --
# see "WHAT IS NOT HERE" below. Keeping the general name means they slot back in
# without moving a prefix that CMAKE_PREFIX_PATH and ld.so.conf both point at.
#
# WHAT IS NOT HERE: rocGRAPH AND hipGRAPH (see issue #126)
# They built, and were dropped anyway, because they cannot be compiled for the
# GPU this image targets. Measured on one 24-core machine, same source, same
# patches, both sharing the CPU:
#
#   GPU_TARGETS=gfx942  (wave64)   135/135 objects, librocgraph.so and
#                                  libhipgraph.so linked, 19 minutes.
#   GPU_TARGETS=gfx1200 (wave32)   stalled at 129/135 after ~2 HOURS, with four
#                                  individual translation units each past 2h and
#                                  still running when it was killed.
#
# That is not "slow", it is unusable: Dockerfile.hip defaults to gfx1200 (a
# Radeon RX 9060 XT) and a GitHub-hosted CI job is capped at 6 hours. It also
# matches what upstream says about itself -- rocGRAPH's default target list is
# wave64 CDNA, its source disables the RDNA3 targets it knows about "because of
# wavefront_size=32", and gfx12 appears in none of its lists.
#
# Getting even that far needed three patches (a LICENSE.md the tree does not
# ship, and two constants ROCm 7 stopped providing as compile-time values). The
# count is the point: a dependency needing that much local repair to build
# against the pinned ROCm is not ready to be one. Issue #126 holds the evidence
# and the cuGraph/hipGRAPH API measurements for when ROCm-DS supports ROCm 7 on
# RDNA.
set -euo pipefail

HIPCOMP_VERSION="${HIPCOMP_VERSION:?HIPCOMP_VERSION must be set, e.g. v2.2.0}"
GPU_TARGETS="${GPU_TARGETS:?GPU_TARGETS must be set, e.g. gfx1200}"
ROCM_DS_PREFIX="${ROCM_DS_PREFIX:-/opt/rocm-ds}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

# --- wave size: the trap this file exists to get right -----------------------
#
# CDNA (gfx9xx) runs wavefronts of 64 lanes; RDNA (gfx10xx/11xx/12xx) runs 32.
# hipCOMP has USE_WARPSIZE_32 for that, OFF by default and documented "e.g.,
# for gfx1100 devices" -- so building it for an RDNA card without this is the
# SILENT WRONG ANSWER case, not a build failure. It is derived from
# GPU_TARGETS, which is the one place in this image that knows the answer.
#
# The single-target check is kept even though hipCOMP alone does not demand it:
# USE_WARPSIZE_32 is one flag for the whole library, so a GPU_TARGETS naming
# both an RDNA and a CDNA card has no correct value to take. Failing here beats
# compiling half-wrong kernels.
case "$GPU_TARGETS" in
  *\;*|*,*)
    echo "install-rocm-ds.sh: GPU_TARGETS='$GPU_TARGETS' lists more than one target." >&2
    echo "  USE_WARPSIZE_32 is one flag for the library; build one target per image." >&2
    exit 1
    ;;
  gfx1*) warpsize_32=ON;  wwr_warp_size=32 ;;
  *)     warpsize_32=OFF; wwr_warp_size=64 ;;
esac
echo "install-rocm-ds.sh: GPU_TARGETS=$GPU_TARGETS -> WWR_WARP_SIZE=$wwr_warp_size"

src=/tmp/rocm-ds-src
trap 'rm -rf "$src"' EXIT
mkdir -p "$src"

# Fetch exactly one ref. `git clone --depth 1` cannot take a SHA, and while
# hipCOMP does have a tag, keeping the by-ref form means an unreleased pin
# needs no new machinery here.
clone_at() {
  local url=$1 ref=$2 dir=$3
  mkdir -p "$dir"
  git -C "$dir" init -q
  git -C "$dir" remote add origin "$url"
  git -C "$dir" fetch -q --depth 1 origin "$ref"
  git -C "$dir" checkout -q FETCH_HEAD
}

# Substitutions are asserted, not hoped for: if upstream changes the text, the
# grep fails the build here rather than leaving a silent no-op that resurfaces
# later as the original error.
patch_one() {
  local file=$1 from=$2 to=$3
  grep -qF "$from" "$file" || {
    echo "install-rocm-ds.sh: expected text not found in $file:" >&2
    echo "  '$from'" >&2
    echo "  Upstream changed; re-check this patch." >&2
    exit 1
  }
  sed -i "s|$(printf '%s' "$from" | sed 's/[][\.*^$/]/\\&/g')|$to|g" "$file"
  grep -qF "$to" "$file" || { echo "install-rocm-ds.sh: patch of $file failed" >&2; exit 1; }
}

clone_at https://github.com/ROCm/hipCOMP-core.git "$HIPCOMP_VERSION" "$src/hipcomp"

# --- hipCOMP pins C++14; ROCm 7.2.4's rocPRIM requires C++17 ------------------
#
# CMakeLists.txt sets CMAKE_CXX_STANDARD and CMAKE_HIP_STANDARD to 14 --
# reasonable for code forked from nvCOMP branch-2.2, and now too old for its own
# dependency:
#
#   rocprim/config.hpp:69: error: "rocPRIM requires at least C++17"
#   ... no template named 'is_same_v' in namespace 'std' ...   (and ~40 more)
#
# It cannot be fixed from the command line: a plain set() shadows the cache
# entry a -D creates, so -DCMAKE_CXX_STANDARD=17 is silently ignored. Patching
# the two lines is the only lever. Raising 14 -> 17 on code this age is low risk
# (the C++17 removals -- auto_ptr, random_shuffle -- are not used here), and it
# is not optional anyway: its dependency demands it.
patch_one "$src/hipcomp/CMakeLists.txt" \
  "set(CMAKE_CXX_STANDARD 14)" "set(CMAKE_CXX_STANDARD 17)"
patch_one "$src/hipcomp/CMakeLists.txt" \
  "set(CMAKE_HIP_STANDARD 14)" "set(CMAKE_HIP_STANDARD 17)"
echo "install-rocm-ds.sh: raised hipCOMP to C++17 (rocPRIM requires it)"

# hipCOMP calls `enable_language(HIP)` and leaves CXX alone, so the
# architecture knob is CMAKE_HIP_ARCHITECTURES. Passing GPU_TARGETS -- the
# convention every other ROCm library here uses -- would be SILENTLY IGNORED,
# leaving the HIP compiler's own default architecture to decide what got built.
cmake -S "$src/hipcomp" -B "$src/hipcomp/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$ROCM_DS_PREFIX" \
  -DCMAKE_PREFIX_PATH="${ROCM_PATH}/lib/cmake" \
  -DCMAKE_HIP_ARCHITECTURES="$GPU_TARGETS" \
  -DUSE_WARPSIZE_32="$warpsize_32" \
  -DBUILD_TESTS=OFF
cmake --build "$src/hipcomp/build" --parallel "$(nproc)"
cmake --install "$src/hipcomp/build"

# Nothing registers this prefix with the loader -- same gap install-rocm.sh
# fills for /opt/rocm/lib, and the same fix.
echo "${ROCM_DS_PREFIX}/lib" > /etc/ld.so.conf.d/rocm-ds.conf
ldconfig

# Record the wave size libhipcomp.so was actually COMPILED with, somewhere
# wwr's own configure can read it.
#
# USE_WARPSIZE_32 is baked into the library here; configuring wwr later with a
# different WWR_WARP_SIZE cannot change it, and the two disagreeing produces no
# error and no warning -- just wrong results from the compression kernels. A
# cmake fragment rather than an env var because it survives `docker run -e`, and
# because including it is one line on the consuming side. The wwr-side check is
# issue #128; writing the file now makes that check a read, not a rebuild.
cat > "${ROCM_DS_PREFIX}/wwr-image-warp-size.cmake" <<EOF
# Generated by docker/install-rocm-ds.sh -- do not edit.
# The wave size hipCOMP in this image was COMPILED for, derived from
# GPU_TARGETS=${GPU_TARGETS}. wwr's WWR_WARP_SIZE must match it.
set(WWR_IMAGE_WARP_SIZE ${wwr_warp_size})
set(WWR_IMAGE_GPU_TARGETS "${GPU_TARGETS}")
EOF

# Findable the way a consumer will look: a CMake config package, and a SONAME
# the loader resolves.
find "$ROCM_DS_PREFIX" -name "hipcomp-config.cmake" -print -quit | grep -q . \
  || { echo "install-rocm-ds.sh: hipcomp-config.cmake not installed" >&2; exit 1; }
ldconfig -p | grep -q "libhipcomp.so" \
  || { echo "install-rocm-ds.sh: libhipcomp.so not on the loader path" >&2; exit 1; }
echo "install-rocm-ds.sh: hipCOMP ${HIPCOMP_VERSION} under ${ROCM_DS_PREFIX}"
du -sh "$ROCM_DS_PREFIX"
