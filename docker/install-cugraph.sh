#!/usr/bin/env bash
# Install cuGraph's C/C++ side (libcugraph + the RAPIDS libraries it links)
# into /opt/rapids, from NVIDIA's PyPI wheels.
#
# Called by docker/Dockerfile.cuda, and inherited (already installed) by
# docker/Dockerfile.combined, which builds on the CUDA image.
#
# WHY A WHEEL, WHEN EVERY OTHER NVIDIA LIBRARY HERE IS APT
# Because there is no apt package. The CUDA repository install-cuda.sh
# registers publishes cutensor, nvcomp, cudss, cusparselt and libnccl -- and no
# cugraph of any spelling. cuGraph is RAPIDS, not the CUDA toolkit, and RAPIDS
# ships conda packages and PyPI wheels only. Conda would mean a second package
# manager and a second Python in an image that already pins one of each, so the
# wheel is the cheaper of the two foreign mechanisms.
#
# The wheel is a C++ SDK with a PyPI wrapper around it, and that is the whole
# reason it is acceptable: libcugraph_cu13 unpacks to headers, .so files and a
# real CMake config package (libcugraph/lib64/cmake/cugraph/cugraph-config.cmake).
# Nothing here imports it from Python.
#
# WHY IT IS NOT IN pyproject.toml / uv.lock
# Dockerfile.base's Python layer says it plainly: dependencies go in
# pyproject.toml, `uv lock`, commit both -- and that rule is right for anything
# the project imports. This is not that. Putting a 515MB CUDA-only wheel in
# pyproject.toml would drag it onto every developer's HOST venv through
# `uv sync`, and into the HIP image, which has no CUDA at all and would resolve
# it anyway. It is a build dependency of the CUDA backend, so it is installed
# where the CUDA backend is built and nowhere else.
#
# --target, not the venv, for the same reason: /opt/venv is the project's
# Python environment and this is not a Python package of the project.
set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:?CUDA_VERSION must be set, e.g. 13-0}"
CUGRAPH_VERSION="${CUGRAPH_VERSION:?CUGRAPH_VERSION must be set, e.g. 26.8.0}"
CUVS_VERSION="${CUVS_VERSION:?CUVS_VERSION must be set, e.g. 26.8.1}"
RAPIDS_PREFIX="${RAPIDS_PREFIX:-/opt/rapids}"

# RAPIDS suffixes its wheels with the CUDA MAJOR (`-cu13`), the same digit
# install-cuda.sh derives for cutensor/nvcomp, and for the same reason:
# CUDA_VERSION is spelled `13-0` for apt and no wheel is named that.
cuda_major="${CUDA_VERSION%%-*}"

# --no-deps IS LOAD-BEARING, not an optimisation. libcugraph's declared
# dependencies include `cuda-toolkit[cublas,curand,cusolver,cusparse,nvrtc]`,
# which are PyPI copies of libraries the apt toolkit already installed a layer
# ago. Letting pip resolve them would add a second libcublas.so to the image,
# of a build nothing here selected, and leave which one a link picks up to
# whichever path came first. So the RAPIDS libraries are named explicitly and
# nothing else is allowed in; CUDA itself comes from apt, as everywhere else in
# this image.
#
# The three companions are not optional -- libcugraph.so links all of them:
#   librmm   RAPIDS memory manager     ~5MB
#   libraft  primitives cuGraph builds on  ~18MB
#   libcuvs  vector search              ~233MB
# with libcugraph itself ~515MB (libcugraph.so 245MB + libcugraph_mg.so 277MB
# multi-GPU + libcugraph_c.so, the C API). ~770MB of wheels, and the single
# largest thing in this image after the toolkit -- it lands in :cuda-ci too,
# which ci.yml pulls on every run. If that becomes the bottleneck, the CUDA
# counterpart of ROCM_PRUNE starts with libcugraph_mg.so.
# --no-cache, measured rather than precautionary: without it uv leaves its wheel
# cache at /root/.cache/uv, and because these wheels are ~770MB compressed that
# cache came to 949MB -- the exact size of the install, doubling this layer to
# 1.95GB for a copy nothing reads again. Layers are additive, so deleting it in
# a later RUN would free nothing; it has to not be written. (Dockerfile.base's
# own `uv sync` leaves 44MB here, which is why the directory exists at all.)
uv pip install \
  --no-cache \
  --python /opt/venv/bin/python \
  --target "$RAPIDS_PREFIX" \
  --no-deps \
  "libcugraph-cu${cuda_major}==${CUGRAPH_VERSION}" \
  "libraft-cu${cuda_major}==${CUGRAPH_VERSION}" \
  "librmm-cu${cuda_major}==${CUGRAPH_VERSION}" \
  "libcuvs-cu${cuda_major}==${CUVS_VERSION}"

# Each wheel unpacks to <prefix>/<name>/{include,lib64} -- ALMOST the shape
# CMake wants from a prefix, and the "almost" cost a debugging session, so:
#
# CMake's config-mode search under a prefix covers <prefix>/lib/cmake/<pkg>/
# but NOT <prefix>/lib64/cmake/<pkg>/ on this image. `lib64` is only searched
# when FIND_LIBRARY_USE_LIB64_PATHS is set, and on Debian multiarch CMake looks
# in lib/<triplet> and lib instead. Measured with --debug-find-pkg=cugraph:
# with /opt/rapids/libcugraph on CMAKE_PREFIX_PATH, the considered list ends at
# `/opt/rapids/libcugraph/cugraph-config.cmake` and never descends into lib64.
# find_path for the headers worked the whole time, which is what makes this
# look like it works until something actually calls find_package.
#
# A `lib -> lib64` symlink per package is the smallest fix: it puts the config
# packages on the path CMake does search, and leaves the wheels' own layout
# (and every absolute path baked into their *-targets.cmake) untouched.
for pkg in libcugraph libraft librmm libcuvs; do
  [ -d "$RAPIDS_PREFIX/$pkg/lib64" ] \
    || { echo "install-cugraph.sh: $pkg did not unpack as expected" >&2; exit 1; }
  ln -sfn lib64 "$RAPIDS_PREFIX/$pkg/lib"
done

# RAPIDS wheels resolve their SIBLINGS at import time, from Python, by
# dlopening them in dependency order -- there is no RPATH from libcugraph.so to
# libraft.so, because the Python package is expected to have loaded it already.
# Nothing here goes through Python, so the loader has to be told instead, or a
# link against cugraph fails on an undefined raft/rmm/cuvs symbol at RUN time
# rather than at build time. The bundled *.libs directories (auditwheel's
# vendored libgomp and friends) are listed for the same reason.
{
  find "$RAPIDS_PREFIX" -maxdepth 2 -type d -name lib64
  find "$RAPIDS_PREFIX" -maxdepth 1 -type d -name '*.libs'
} > /etc/ld.so.conf.d/rapids.conf
ldconfig

test -f "$RAPIDS_PREFIX/libcugraph/lib64/cmake/cugraph/cugraph-config.cmake"
test -f "$RAPIDS_PREFIX/libcugraph/include/cugraph_c/graph.h"
# Through the symlink too -- this is the path CMake will actually take, and
# asserting only the lib64 one is what let the find_package gap ship the first
# time.
test -f "$RAPIDS_PREFIX/libcugraph/lib/cmake/cugraph/cugraph-config.cmake"
ldconfig -p | grep -q 'libcugraph\.so' \
  || { echo "install-cugraph.sh: libcugraph.so not on the loader path" >&2; exit 1; }
echo "install-cugraph.sh: cuGraph ${CUGRAPH_VERSION} under ${RAPIDS_PREFIX}"
du -sh "$RAPIDS_PREFIX"
