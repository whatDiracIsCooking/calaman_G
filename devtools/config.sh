#!/usr/bin/env bash
# Per-project settings for everything in devtools/. EDIT THIS FILE, not the
# scripts -- the scripts are meant to survive being copied into the next repo
# unchanged, and this is the one place that knows what the project is called
# and how its tests run.
#
# Sourced (never executed) by devcontainer.sh, worktree.sh, doctor.sh,
# cpp-tier.sh and prepush-tests.sh. Every value uses ${VAR:-default} so an
# environment variable still wins for a one-off:
#
#   JOBS=16 devtools/prepush-tests.sh
#
# The one thing this file CANNOT reach is the .devcontainer/<variant>/devcontainer.json
# files -- JSON cannot source shell. The volume names, the container name and the
# .git bind path are spelled out there too; PROJECT_NAME below must match the
# `<name>-pytest-tmp-` style prefixes used there, or `worktree.sh rm` and
# `worktree.sh gc` will not recognise this project's volumes as its own.
# doctor.sh checks that the two agree and warns when they have drifted.
#
# The C++ side has a second config file with the same job: CMakePresets.json
# holds the build settings (compiler, build type, GPU architecture, build
# directory). What lives HERE is only which preset the scripts should drive --
# see CMAKE_PRESET / CTEST_PRESET below.

# --- identity -------------------------------------------------------------

# Prefix for this project's docker volumes and per-worktree build images, and
# the name of the devcontainer. Must match devcontainer.json (see above).
# It ends up in docker resource names, so keep it lowercase.
PROJECT_NAME=${PROJECT_NAME:-gpumod}

# Which devcontainer.json devcontainer.sh drives. Relative paths resolve
# against the repo root, so this works from any cwd and from any worktree.
# There is one per GPU file of docker/ (Dockerfile.cuda / .hip / .combined):
#
#   .devcontainer/cuda/devcontainer.json     THE development container:
#                                            clang-20 + libc++, CMake 4.2,
#                                            CUDA 13. Needs an NVIDIA GPU and
#                                            nvidia-container-toolkit.
#   .devcontainer/hip/devcontainer.json      the same toolchain with ROCm and
#                                            no CUDA at all. Needs an AMD card
#                                            and the amdgpu kernel driver. It
#                                            pins CMAKE_PRESET/CTEST_PRESET to
#                                            `hip` in containerEnv, so
#                                            cpp-tier.sh needs no flag in there.
#   .devcontainer/combined/devcontainer.json both SDKs, ~40GB. For working on
#                                            both backends in one shell;
#                                            stays on the `default` (CUDA)
#                                            preset, `--preset hip` for the
#                                            other.
#
# This is only the FALLBACK. For one call, use the variant flag instead --
# `devtools/devcontainer.sh --hip up` -- and for a whole shell session, export
# the variable. Precedence is flag, then variable, then this.
#
# Points at the CUDA variant because that is where the project actually builds.
# Each worktree sources its OWN copy of this file (lib.sh resolves the repo
# root from its own location), so changing the line below switches the default
# for that checkout alone -- useful for a worktree dedicated to one backend,
# as long as you do not commit it.
#
# Containers are keyed on the workspace folder AND this path, so the variants
# of one worktree are separate containers and can be up at once. They share
# that worktree's volumes, so run the suite in one at a time.
DEVCONTAINER_CONFIG=${DEVCONTAINER_CONFIG:-.devcontainer/cuda/devcontainer.json}

# --- C++ build (CMake) ------------------------------------------------------

# Which CMakePresets.json presets devtools/cpp-tier.sh drives. The presets
# themselves -- compiler, build type, CUDA arch, build directory -- live in
# CMakePresets.json; only the CHOICE lives here.
#
#   default     Release, clang, CUDA_ARCHITECTURES=native, into build/
#   workstation the SAME configuration into build-workstation/, so a host build
#               and a container build can coexist without reconfiguring each
#               other. Nothing is prebuilt for either; GoogleTest is fetched.
#   debug / asan / compute-sanitizer
#   hip         the ROCm backend (build-hip/)
#   compile-time  builds the static_assert tier only: no GPU, no GoogleTest
#   ci-cuda     what .github/workflows/ci.yml builds: the full CUDA tree with
#               the architecture PINNED, and a test preset that excludes the
#               `gpu` label. Runnable here too, and worth it to reproduce a CI
#               failure exactly -- `devtools/cpp-tier.sh --preset ci-cuda`
#               gives the same selection on this box that the runner gets.
#   ci-hip      the HIP half, deliberately the SAME SHAPE: the whole tree
#               through clang's -x hip front end, runtime binaries included,
#               then the same -LE gpu. Compiling the test sources is the
#               point -- clang is the stricter front end. No device either.
#
# Read either one as compile-and-link plus a thin runtime slice, NOT as a test
# of GPU behaviour: `-LE gpu` leaves two CPU-only conversion suites that
# actually compute, one load check per device binary, and the link checks.
# The four dispatch checks run at build time on both, so they are covered
# without appearing in the ctest count.
#
# ci-cuda pins CMAKE_CUDA_ARCHITECTURES to 86 rather than widening it, so that
# CI compiles exactly what `native` compiles on the reference box and a red run
# reproduces locally with one command.
#
# THERE ARE NO PER-ARCHITECTURE PRESETS. `base` uses native and ci-cuda pins
# 86; between them that is every case this project has, and sm_86 is what the
# reference box, docker/Dockerfile.cuda's CUDA_ARCH and ci-cuda all already
# say. volta (70), ampere (80), hopper (90) and portable (70;80;90) were
# removed rather than corrected -- nothing exercised them and two could not
# configure at all. Target something else with a -D and a build dir of its own:
#
#   cmake --preset default -B build-h100 -DCMAKE_CUDA_ARCHITECTURES="90"
#
# It has to be -D. That overrides a preset's cacheVariables, where an
# environment variable does not: CUDAARCHS is consulted only when the cache
# variable is unset, and `base` always sets it.
#
# NOTHING MAY GO BELOW sm_75. CUDA 13's nvcc floor is compute_75, so
# CMAKE_CUDA_ARCHITECTURES=70 dies at CONFIGURE time with `nvcc fatal :
# Unsupported gpu architecture 'compute_70'` -- reported as CMake's "Check for
# working CUDA compiler - broken", which is why volta and portable sat broken
# through the whole CUDA 13 bump without anyone noticing. A pre-Turing card
# needs a 12.x CUDA_VERSION in docker/Dockerfile.cuda, whose header has the
# rest of it.
#
# Override for one run rather than editing:
#   CMAKE_PRESET=asan CTEST_PRESET=asan devtools/cpp-tier.sh
#
# CTEST_PRESET is separate because not every configure preset has a matching
# test preset -- only default, workstation, debug, asan, hip, compile-time,
# ci-cuda, ci-hip and coverage do. Set it empty to configure and build without
# running ctest.
CMAKE_PRESET=${CMAKE_PRESET:-default}
CTEST_PRESET=${CTEST_PRESET:-default}

# Which preset devtools/coverage.sh configures, builds and runs ctest under to
# collect clang source-based coverage. Its own build directory (build-coverage/)
# and WWR_COVERAGE=ON live in CMakePresets.json; only the CHOICE lives here.
# The report and merged .profdata are written under that build dir, so they are
# covered by the /build*/ line in .gitignore and need no cleanup of their own.
COVERAGE_PRESET=${COVERAGE_PRESET:-coverage}

# Source files devtools/coverage.sh drops from its llvm-cov summary, as an
# -ignore-filename-regex over the full path. Coverage here measures what the
# RUNTIME ctest suites reach, but two kinds of src/ file are verified at COMPILE
# time instead and never execute, so counting them only drags the number down
# with lines that are already tested a different way:
#
#   * src/cuda/*, src/hip/*  -- the vendor raw-module re-exports, which validate
#     and re-declare the SDK headers at import time; there is no runtime surface.
#   * src/{blas,complex,fft,rand,solver,sparse}.cppm -- the backend-neutral gpu*
#     wrapper layer, whose one-line token-paste forwards are proved by the
#     dispatch checks (test/shared/dispatch.py) and static_asserts, and were
#     deliberately never given runtime suites (see CLAUDE.md, "Two kinds of C++
#     test"). fp16/bf16/runtime_api.cppm are NOT here: they carry host-side logic
#     the runtime suites do execute, so they stay in the denominator.
#
# Set empty to report over all of src/. Keep this in sync with the wrapper set
# under src/ if a neutral module is added or removed.
COVERAGE_IGNORE_REGEX=${COVERAGE_IGNORE_REGEX:-'/src/(cuda|hip)/|/src/(blas|complex|fft|rand|solver|sparse)\.cppm$'}

# --- cross-backend check --------------------------------------------------

# devtools/cross-backend-check.sh compiles the tree for the backend this build
# is NOT targeting. It exists because the two backends go through DIFFERENT
# FRONT ENDS -- a .cu is nvcc's problem under CUDA and clang's under HIP -- and
# nvcc is the more permissive of the two, so a whole class of error is invisible
# until someone builds the other side -- a functor with a `const` member of
# class type, say, which clang rejects for parallel_for's device_functor concept
# while nvcc waves it through.
#
# A build targets exactly one backend and CMAKE_PRESET above picks it, so these
# describe the OTHER one. Flip all five together if the project's default
# backend ever becomes HIP.

# Which backend to check, its preset, and the tool whose presence means this
# machine can do it without a container. CROSS_CHECK_PRESET must name a preset
# whose binaryDir differs from CROSS_CHECK_BUILD_DIR's use below -- the script
# passes -B explicitly, so the `hip` preset's own build-hip/ is left alone and
# the two never reconfigure each other.
CROSS_CHECK_BACKEND=${CROSS_CHECK_BACKEND:-HIP}
CROSS_CHECK_PRESET=${CROSS_CHECK_PRESET:-hip}
CROSS_CHECK_TOOL=${CROSS_CHECK_TOOL:-hipconfig}

# The image carrying that toolchain, used when this machine has no hipconfig.
# `docker/build.sh hip` builds and tags it -- that script derives the tag from
# PROJECT_NAME exactly as the line below does, so the two agree by construction
# rather than by convention; the script prints the build command when it is
# absent.
CROSS_CHECK_IMAGE=${CROSS_CHECK_IMAGE:-${PROJECT_NAME}:hip}

# Its own build directory, separate from every preset's, because two presets
# sharing one directory silently reconfigure it back and forth -- a full
# rebuild each way. Matches the `build*/` line in .gitignore.
CROSS_CHECK_BUILD_DIR=${CROSS_CHECK_BUILD_DIR:-build-cross-check}

# The device-kernel libraries, one per line -- what --device-only builds when
# you want the 3s check instead of the 11s one. These are the targets holding
# .cu sources, i.e. every wwr_add_gpu_device_library() call site. A module
# added with a .cu and left out of this list is still covered by the default
# (whole-tier) run; it only loses the narrow mode. An entry naming a target that
# does NOT exist, on the other hand, fails the whole --device-only run with
# ninja's `unknown target`. Keep this list matched to test/gpu/CMakeLists.txt;
# nothing checks it automatically.
CROSS_CHECK_DEVICE_TARGETS=${CROSS_CHECK_DEVICE_TARGETS:-"wwr.test.gpu.cooperative_groups.device
wwr.test.gpu.wmma.device
wwr.test.gpu.complex.device
wwr.test.gpu.fp16.device
wwr.test.gpu.bf16.device
wwr.test.gpu.atomics.device"}

# --- the ROCm runtime tier -------------------------------------------------
#
# cross-backend-check.sh above proves the other backend COMPILES. On a machine
# with a real AMD card, `cpp-tier.sh --rocm` is the stronger statement: it runs
# that backend's ctest against the hardware, by re-execing cpp-tier.sh inside
# ROCM_IMAGE with the device nodes and group ids below.
#
# It exists because getting that passthrough right is fiddly and its failure
# mode is indistinguishable from having no card at all -- see ROCM_GROUPS.

# The image with the ROCm toolchain, and the preset to run in it. Same image
# cross-backend-check.sh uses; named separately so a machine can point the two
# at different builds without one silently following the other.
ROCM_IMAGE=${ROCM_IMAGE:-${PROJECT_NAME}:hip}
ROCM_PRESET=${ROCM_PRESET:-hip}

# The device nodes ROCm needs, one per line. /dev/kfd is the compute driver
# interface and /dev/dri carries the render nodes; both are required, and
# neither is passed into a container by default the way --gpus all handles
# NVIDIA.
ROCM_DEVICES=${ROCM_DEVICES:-"/dev/kfd
/dev/dri"}

# The HOST groups owning those nodes, one per line. Both container paths
# resolve these to NUMERIC gids with getent before passing them to
# --group-add, and that matters more than it looks: `--group-add render`
# resolves the name INSIDE the container, where it has a different id or does
# not exist, so the run then fails at the first device call with
# "hipErrorNoDevice (no ROCm-capable device is detected)" -- which reads
# exactly like a box with no GPU, and has been misread as one.
#
# cpp-tier.sh --rocm builds its own --group-add args from this list, and fails
# loudly when a name does not resolve. devcontainer.sh instead exports one
# WWR_<NAME>_GID per entry (render -> WWR_RENDER_GID) for the
# ${localEnv:...} references in the hip/combined runArgs, and only warns: the
# json carries a fallback gid so that opening it WITHOUT this script still
# passes something. Adding a name here reaches the first automatically; the
# second also needs the reference written into the json, since JSON cannot
# loop.
ROCM_GROUPS=${ROCM_GROUPS:-"render
video"}

# Parallel compile jobs for `cmake --build`. Distinct from JOBS below, which is
# the pytest worker count: a C++23 module build is memory-hungry per job (the
# scanner plus a BMI cache), so the right number is usually LOWER than the core
# count, and lower still than what you would give pytest. Empty lets Ninja pick
# (cores + 2), which is what OOM-kills a 16-core box on a module-heavy tree.
BUILD_JOBS=${BUILD_JOBS:-}

# --- tests ----------------------------------------------------------------

# How the test suite is invoked, as an array so arguments with spaces survive.
# Anything goes here -- pytest, go test, cargo test, make check -- as long as
# it takes extra arguments on the end and exits non-zero on failure. Scripts
# that need a pytest specifically (prepush) resolve it through VENV_PATHS
# below instead, so change both if you leave Python behind.
TEST_CMD=${TEST_CMD:-pytest}

# Default xdist worker count. 8 rather than `auto` on purpose: several worktree
# containers can be up at once, and `auto` in each would claim every core on
# the box. Raise it for a single-checkout project on a big machine. Set to
# empty to drop the -n flag entirely (a suite with no pytest-xdist).
JOBS=${JOBS:-8}

# Host CPU bounds for a devcontainer, applied by devcontainer.sh on `up` and
# `rebuild`. Both empty = unbounded, and nothing is applied.
#
#   CPUSET  pin the container to specific host cores, e.g. 0-11
#   CPUS    cap it at n cores' worth of CPU quota, e.g. 8
#
# These bound EVERYTHING the container runs -- a compiler, a linker, a build
# system, anything a shell inside it starts -- where JOBS above only bounds the
# test runner. Two worktree containers pinned to disjoint cores (CPUSET=0-11
# here, CPUSET=12-23 there) physically cannot contend, which is what makes a
# timing measurement in one of them mean anything. A container is per workspace
# folder, so these are per worktree: two `shell`s from the same checkout share
# one container and one set of limits.
#
# Left empty here by default because the right values are machine-specific --
# set them per invocation (CPUSET=0-11 devtools/devcontainer.sh up), or write
# them in here once the box this project builds on is settled.
CPUSET=${CPUSET:-}
CPUS=${CPUS:-}

# Extra marker expression for the default run. EMPTY: the Python suite is a
# handful of fast files with no expensive tier behind them, so there is nothing
# to hold back. Add a marker here only alongside something that runs a slower
# half.
# NB if you do set one, single-quote it: inside ${VAR:-default} bash strips one
# level of quoting, so an unquoted `-m "not gpu"` would reach pytest as the
# three words `-m`, `not`, `gpu` -- a marker expression of `not` and a path
# called `gpu`, which collects nothing and says only "no tests ran".
FAST_TEST_ARGS=${FAST_TEST_ARGS:-}

# What the pre-push gate runs. Deliberately a curated list rather than the
# whole suite: the gate should cost seconds, so anything driving a compiler, a
# GPU or the network belongs in CI instead. The value is passed to pytest
# verbatim after FAST_TEST_ARGS, so paths, node ids and -k expressions all
# work. Empty means "whatever pytest collects by default", which is the right
# starting point for a small project.
PREPUSH_PATHS=${PREPUSH_PATHS:-}

# Where cpp-tier.sh writes its run logs. Relative paths are resolved against the
# repo root. Gitignored by default. `.gitignore`, `.dockerignore` and the CI
# artifact upload all name this path, so keep them in step if you rename it.
#
# NB docker/compose.yaml writes its own logs to .log/ instead -- it tees
# configure/build/test output there, and is not driven by this file at all.
SLOW_TIER_REPORTS=${SLOW_TIER_REPORTS:-.slow-tier-reports}

# --- environment ----------------------------------------------------------

# Where to look for the project's virtualenv, in order; the first entry with a
# bin/pytest wins. `.venv` is what `uv sync` makes on the host, `/opt/venv` is
# what the Dockerfile bakes into the image.
VENV_PATHS=${VENV_PATHS:-.venv /opt/venv}

# Optional tools doctor.sh reports on, one `name:what is lost without it` per
# line. A missing one is a WARN (a slice of the suite will skip), never a FAIL.
#
# Most of this list is absent on the HOST and present in the CUDA container,
# which is the point: run doctor.sh on both sides and the warnings tell you
# which half of the workflow you are on. A host with none of the C++ toolchain
# is a perfectly good place to edit, lint and run the Python suite.
DOCTOR_OPTIONAL_TOOLS=${DOCTOR_OPTIONAL_TOOLS:-"docker:devcontainer, compose services and worktree containers
gh:the PR flow (also needs GH_TOKEN)
cmake:configuring and building the C++ tree at all
ninja:the generator every preset uses
clang++:the C++23 module build (must be clang -- CMakeLists.txt refuses gcc)
clang-scan-deps:module dependency scanning; a FATAL_ERROR at configure without it
nvcc:compiling .cu translation units and the whole CUDA backend (src/cuda)
hipconfig:the HIP backend (src/hip); present only in the \`hip\`/\`combined\` image
llvm-objdump:test/extension/build_time's blas dispatch check
llvm-cxxfilt:test/extension/build_time's blas dispatch check
compute-sanitizer:the GPU memcheck/racecheck run (docker compose run --rm compute-sanitizer)
nsys:Nsight Systems profiling from inside the container
ccache:warm rebuilds; without it every configure recompiles from scratch
clang-tidy:the advisory lint pass (cpp-tier.sh --tidy, and CI's cuda leg)
clang-format:the C++ formatting pass (see CLAUDE.md -- it is not a git hook)
cmake-format:the CMake formatting pass; run by hand, not by any hook
cmake-lint:the CMake lint pass CI runs on every PR"}

# Which GPU vendors doctor.sh probes for a LIVE DEVICE, space-separated:
# `nvidia`, `amd`, or empty to drop the section entirely (a CPU-only repo).
#
# This is about the DEVICE, not the toolchain -- the two are independent and
# both matter. nvcc present with no card still cannot configure the `default`
# preset, because CMAKE_CUDA_ARCHITECTURES=native asks the driver what is
# installed at CONFIGURE time; and a card present with no toolchain is the
# normal host state. Nothing in test/ calls GTEST_SKIP, so a device-less box
# FAILS the runtime tests rather than skipping them, which is the single
# easiest red run to misread as a broken change (see CLAUDE.md). Both vendors
# are listed because this box can have both and a build targets exactly one.
DOCTOR_GPU_VENDORS=${DOCTOR_GPU_VENDORS:-"nvidia amd"}

# Paths that must exist for some slice of the suite to run, one
# `path:what skips without it` per line -- typically submodules or fixture
# trees.
#
# Deliberately empty. The project's only C++ dependency is GoogleTest, which
# deps/CMakeLists.txt fetches and builds from source at configure time, so no
# preset depends on anything being prebuilt in the image. Do not add a path here
# unless a build actually reads it.
DOCTOR_REQUIRED_PATHS=${DOCTOR_REQUIRED_PATHS:-""}
