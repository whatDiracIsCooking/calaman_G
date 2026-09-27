# Devcontainers

One per GPU file of `../docker/`.

| File | What it is |
|---|---|
| `cuda/devcontainer.json` | **The development container.** `../../docker/Dockerfile.cuda` plus `--gpus all`: clang-20 + libc++, CMake 4.2, Ninja, CUDA 13, ccache. The default. GoogleTest is fetched at configure time; nothing is prebuilt in `/opt`. |
| `hip/devcontainer.json` | The AMD variant: `Dockerfile.hip` — the same toolchain with ROCm and **no CUDA at all**. Pins `CMAKE_PRESET`/`CTEST_PRESET` to `hip`, so a bare `devtools/cpp-tier.sh` inside it builds the HIP backend into `build-hip/`. |
| `combined/devcontainer.json` | `Dockerfile.combined`: both SDKs, ~40GB. For working on both backends in one shell. No preset pin — `default` is CUDA, `--preset hip` is the other. |

**Each one carries an `initializeCommand` that builds its parent image first.**
The four Dockerfiles chain by tag (`FROM ${PARENT_IMAGE}`), and a devcontainer
build is a single `docker build` with no way to produce that parent — so the hook
runs `../docker/build.sh` on the host before the image is built. Without it the
build fails on the first line with `pull access denied for gpumod`, which reads
like a registry problem. See `../docker/Dockerfile.base`'s header.

> **`devcontainer build` does not run `initializeCommand`; only `up` does.**
> Verified against @devcontainers/cli 0.89.0: `up` runs the hook as its very
> first step, before it resolves the image at all, but a bare
> `devcontainer build --workspace-folder .` skips lifecycle hooks and dies with
> `Command failed: docker pull gpumod:base`. Nothing in `devtools/` uses
> `build` — `devcontainer.sh` only ever calls `up`, `exec` and `down` — so the
> supported paths are unaffected. If you want just the image, use
> `docker/build.sh cuda`, which is what the hook calls anyway.
| `post-create.sh` | The `postCreateCommand` for all three variants, written once: seed the Claude config, trust the checkout, init submodules, install the pre-commit hook (leaving a host-installed one alone), and wire `gh` into git. JSON cannot share a fragment, so each `devcontainer.json` just calls this. |
| `seed-claude-config.sh` | Copies the agent's host config into the container once, at create time. Invoked by `post-create.sh`. |

## Handing each one its GPU

The three GPU variants differ in `"build".target` and in `runArgs`, and the
`runArgs` half is not symmetric:

- **NVIDIA** is one flag, `--gpus all`, read by the NVIDIA container runtime.
  It works because the `cuda` stage sets `NVIDIA_VISIBLE_DEVICES` and
  `NVIDIA_DRIVER_CAPABILITIES` explicitly — the `nvidia/cuda` base images set
  those, and this tree builds on plain Ubuntu instead.
- **AMD** has no equivalent flag. The driver is two device nodes, passed in
  with `--device=/dev/kfd --device=/dev/dri`, and the container user must be in
  the groups that own them — hence the two `--group-add` args, which carry
  **numeric host gids** (`getent group render video`). A *name* there is the
  trap: `--group-add render` resolves inside the container, where
  `docker/install-rocm.sh` made `render` 110, while the bind-mounted nodes keep
  their host ownership — so the user joins a group the device does not grant
  and the first device call fails with `hipErrorNoDevice`, which reads as a box
  with no AMD card. Do not rebuild the image to chase a mismatch.
  `seccomp=unconfined` is ROCm's documented requirement; without it some HSA
  queue paths fail as `HSA_STATUS_ERROR_OUT_OF_RESOURCES` rather than as a
  permission error.

**You should not have to edit those gids.** They are written as
`${localEnv:WWR_RENDER_GID:109}`: `devtools/devcontainer.sh up|rebuild`
resolves `ROCM_GROUPS` (`devtools/config.sh`) with `getent` and exports one
`WWR_<NAME>_GID` per group, so any host gets its own host's ids — the same
resolution `cpp-tier.sh --rocm` does. The literal after the colon is only the
fallback for opening a config **without** that script (a VS Code "Reopen in
Container", a bare `devcontainer up`); it is a last-known-good host gid rather
than a group name, because a name is the silent-failure case above. Either
export the variable yourself for that path, or update the fallback.

`combined` carries both sets. On a host with only one vendor's plumbing it
still comes up, and only that *other* backend's **runtime** tests fail —
compile tests are unaffected, since both toolchains target an architecture
rather than the card present. Note they fail rather than skip (see CLAUDE.md).

## Which one you get

Three ways, highest precedence first.

**A flag before the command** — the everyday way. It names a directory under
`.devcontainer/`:

```bash
devtools/devcontainer.sh --hip shell
devtools/devcontainer.sh --combined rebuild
devtools/devcontainer.sh --cuda up          # the default; explicit is fine
```

It must come *before* the command, since everything after `shell` and `test` is
passed through to what runs inside. `devtools/worktree.sh add <name> --hip --up`
forwards it, so a new worktree can come up on any backend in one command.

The mapping is by convention, not from a list, so adding a
`.devcontainer/<name>/` makes `--<name>` work with no edit to either script. A
`--flag` that doesn't name an existing config is reported as unknown rather
than silently treated as a variant, and two variant flags in one call is an
error rather than last-one-wins.

**`DEVCONTAINER_CONFIG`** — still the way to point at a config outside this
layout, and the way to make a whole shell session use one variant:

```bash
export DEVCONTAINER_CONFIG=.devcontainer/hip/devcontainer.json
```

A relative path is resolved against the repo root, so it works from any cwd and
in any worktree.

**`devtools/config.sh`** holds the fallback (the CUDA variant). Each worktree
sources its *own* copy — `lib.sh` resolves the repo root from its own location
— so editing it changes the default for that checkout alone. Handy for a
worktree dedicated to one backend; just don't commit it.

`devtools/doctor.sh` prints the active config, and **FAILs** if it points at a
file that is not there — every `devcontainer.sh` command refuses until that is
fixed. Its `PROJECT_NAME`-drift check globs every `devcontainer.json` in here, so a new
variant is covered without editing `doctor.sh`.

### Several variants at once

A container is keyed on the workspace folder **and** the config file — the CLI
labels each with `devcontainer.local_folder` and `devcontainer.config_file` —
so the cuda, hip and combined containers for one worktree are three separate
containers that can be up simultaneously. `devcontainer.sh` matches on both
labels, so `down` and the `CPUSET`/`CPUS` limits act on the variant you
selected and leave the siblings alone.

**That cuts both ways: the variant flag is needed on teardown too.** A `--hip
up` followed by a bare `down` takes down the *CUDA* container and leaves the
HIP one running. When you have lost track:

```bash
devtools/devcontainer.sh down --all
```

`devtools/worktree.sh rm <name>` has no such hazard — it matches on the
workspace folder alone and removes every variant's container and image for that
worktree, precisely so teardown never depends on remembering which one you had
up.

They **share** that worktree's three named volumes, deliberately:
`devtools/worktree.sh` reconstructs volume names as
`<PROJECT_NAME>-<suffix>-<worktree-basename>` and treats anything whose tail is
not a live worktree name as an orphan to delete, so a per-variant suffix would
make `worktree.sh gc` eat them. The consequence: two variants of the same
worktree share one `/tmp/pytest-of-ubuntu`, and pytest's keep-the-last-three
numbered-dir cleanup in one is then free to delete a run the other is using.
**Run the suite in one variant at a time.**

Note that the bare `devcontainer` CLI and any IDE "Reopen in Container" action
look only for a top-level `.devcontainer/devcontainer.json`, which this repo no
longer has. Point the IDE at `cuda/devcontainer.json` explicitly (or one of the
other variants), or it will not find a config to open.

## Per-project edits

All three JSON files carry `// EDIT` markers on the three things a new project
must change. **A rename has to touch all three, in every file** — this list is
what to check by hand.

1. `"name"` — what the container is called.
2. The `PROJECT_NAME` build arg — stamped on the image as a label, which is how
   `devtools/worktree.sh gc` recognises this project's build images and leaves
   every other repo's alone.
3. The three `source=gpumod-…` volume names — `worktree.sh rm` and `gc`
   reconstruct these from `PROJECT_NAME` in `devtools/config.sh`, so the two
   must agree. `devtools/doctor.sh` checks each JSON file separately and names
   the one that drifted; while they disagree, every worktree you tear down leaks
   its whole set of volumes.
The **`.git` bind mount** used to be a fourth. It is now injected: the mount
source reads `${localEnv:WWR_GIT_DIR}`, and `devtools/devcontainer.sh`
resolves your main checkout's `.git` common dir with `git rev-parse` and exports
it on `up`/`rebuild`, so no host path is written into the tracked JSON. A
worktree's `.git` is a *file* pointing into `.git/worktrees/<name>`, outside the
workspace folder, and without the mount every git command inside a worktree
container fails with "not a git repository", taking `postCreateCommand` — and
with it the submodule init and the pre-commit hook install — down with it. The
literal after the colon (`…:/absolute/path/to/your-root/main/.git`) is only a
fallback for opening the config directly (VS Code "Reopen in Container"), which
does not run `devcontainer.sh`; edit that to your own `.git` if you work that
way.

JSON cannot source shell, which is why (2) and (3) are duplicated here rather
than read from `devtools/config.sh`.

## Rebuilding

```bash
devtools/devcontainer.sh rebuild
```

Use `rebuild`, not `up`, after editing either JSON file or any
`../docker/Dockerfile.*`: `up`
reuses the running container, applies none of the change, and reports success
with the same container id. The workspace is a bind mount and the caches are
named volumes, so both survive a rebuild.

A `Dockerfile.base` edit needs nothing extra: the `initializeCommand` rebuilds
the parent on every `up` and `rebuild` rather than skipping when the tag already
exists, so the retagged parent is what the child then builds on. That is
deliberate — skipping on tag presence is how a child would silently keep a stale
toolchain.

The CUDA image is large and its first build is long — LLVM 20 from
apt.llvm.org, CMake and Ninja from tarballs, and the CUDA toolkit from apt. The
HIP and combined images are longer still: ROCm is ~30GB installed. Subsequent
rebuilds hit the layer cache unless you changed something early in the file,
and the `base` stage is shared, so the second GPU variant you build skips the
whole toolchain half.

No C++ dependency is baked into any of them. GoogleTest — the only one — is
fetched and built by `deps/CMakeLists.txt` at configure time.

## Lockfiles

Each variant has a `devcontainer-lock.json` beside its `devcontainer.json`, and
**all three are tracked.** They pin the two `features` every variant declares
(`claude-code`, `github-cli`) to an exact version and digest, the way `uv.lock`
pins the Python side.

Tracking them is what keeps `devtools/devcontainer.sh` from dirtying the tree.
Verified against the devcontainer CLI 0.89.0, both directions:

- **No lockfile** — the CLI resolves the features to whatever is newest and
  **writes the file**. That is where a stray untracked
  `.devcontainer/<variant>/devcontainer-lock.json` comes from: a `build`/`up`
  on a variant that had none.
- **Lockfile present** — the CLI honours it and writes nothing. A `build` with
  a lockfile pinned to `github-cli` 1.1.1 left it at 1.1.1 with the worktree
  clean, even though 1.1.2 was available.

So gitignoring them would not have stopped the regeneration, only hidden it —
and every rebuild would silently drift to the newest feature version. Tracking
stops it at the source.

To refresh one deliberately, use the CLI rather than deleting the file (which
re-pins to newest as a side effect of the next build):

```bash
npx -y @devcontainers/cli outdated --workspace-folder . \
  --config .devcontainer/cuda/devcontainer.json --output-format text
npx -y @devcontainers/cli upgrade  --workspace-folder . \
  --config .devcontainer/cuda/devcontainer.json
```

`upgrade` rewrites the lockfile from the registry without building an image, so
it works for `hip` and `combined` without paying for a ~20-40GB build. Commit
the result like any other lockfile bump.

The three are pinned independently and need not agree. They are alternative
environments, not one environment, so a variant is bumped when there is a reason
to bump it.

## Bounding it

`CPUSET` (host cores) and `CPUS` (a quota) in `devtools/config.sh` are applied by
`up` and `rebuild` via `docker update` — live, no restart. They bound everything
in the container, unlike `JOBS` (pytest workers) and `BUILD_JOBS`
(`cmake --build` jobs), which bound only their own runner. Two worktree
containers pinned to disjoint cores are the only way concurrent worktrees stop
stealing each other's timings.
