---
name: docstyle
description: >-
  This repo's source-documentation convention — what belongs in a file header, a
  declaration comment, a CMake file, a README, or docs/architecture.md, and the
  budget each gets. Use when writing or reviewing comments and doc blocks
  under src/ or cmake/, when a file header or a CMakeLists.txt comment has grown
  into an essay, or when the user asks about documentation style, verbosity,
  comment bloat, or where a piece of rationale should live. Not for user-facing
  docs or skills — that's `audit`.
---

# Documentation conventions

A doc comment states the **contract**: what a name is, what it takes, what it
guarantees, what constrains its use. It is not a lab notebook and not a
tutorial. Rationale, evidence, and history are all legitimate — they just live
somewhere else, and this skill says where.

`src/`'s C++ module and header files (`.cppm`, `.cuh`, `.h`) were converted to
this convention wholesale once (see the `docstyle` commits). Those now hold:
every file header is within budget, and the provenance idiom appears in none of
them. Keep it that way per file as you touch it — **do not re-run a sweep** on
those. The `.cpp`/`.cu` sources and the CMake files (`cmake/`, `CMakeLists.txt`)
were **not** part of that sweep and are not yet within budget — convert those as
you touch them (see "CMake files" and "Scope" below).

## The four homes

Every sentence you are about to write answers one of four questions. Put it
where that question is answered.

| The sentence answers | Home | Form |
|---|---|---|
| What is this name, how is it called, what must hold? | the declaration's own `///` block | Doxygen tags |
| What is in this file, and what invariant governs it? | the file header | ≤25 lines of prose |
| Why is this *directory* shaped this way? | that directory's `README.md` | prose, stated once |
| Why did **we** choose this, or why does the **vendor**/language force it? | `docs/architecture.md` | numbered section, dated |
| What changed, and from what? | the commit message / PR body | — |

## The budget

**25 lines of prose**, `/**` to `*/`. Two things do not count, because a header
*is* reference material and these are reference, not verbosity:

- a `Usage:` example block
- a Markdown table

That distinction is the whole rule — prose is budgeted, reference is not. It is
not a licence to reformat an essay as a table.

The number came from applying this to all 111 files in `src/`: 20 was too tight
the moment a header carried a runnable example, and chasing it produced worse
prose, not shorter. 25 still catches 36 of the 40 files that were over.
`src/cuda` sat inside it already, at a median of ~13.

That measurement is over C++ `/**` headers; the same 25-line budget applies to a
CMake file's leading `#` block by the same principle — constraint is budgeted,
story moves out — not by a separate count.

Also budgeted: **a declaration block is 5 lines**, plus one `@param`/`@tparam`
line each. And **one line per constraint** — if a constraint needs a paragraph,
the paragraph belongs in `docs/architecture.md`, and the header gets the line
plus the link.

## The file-header template

```cpp
/**
 * @file <name>
 * @brief <one line, under 80 columns>
 *
 * <1-3 sentences: what this file provides and who includes/imports it.>
 *
 * <Optional: constraints, one line each. Link out for the why.>
 *
 * Usage:
 *   import gpumod.<x>;
 *
 *   <two or three lines of real calling code>
 */
```

No other sections. No `-----` rules inside the block.

`src/gpu_backend.h` is the model for a file that genuinely has a contract:
its three macros with a one-line contract each, the one non-obvious constraint
("a function reference carries no default arguments, and cannot name an
overload set"), and stop.

## Banned constructs

**Verification provenance.** Delete it from headers entirely.

> `-- confirmed by direct compilation with this project's exact hip::device`
> `flags (-x hip --offload-arch=gfx1200), not assumed.`

A header describes code as it is *now*. "Verified on ROCm 7.2.4" asserts a fact
about a toolchain, dated nowhere and re-checked by nobody — the reader cannot
tell whether it still holds, and it silently rots. Dated in
`docs/architecture.md`, the same sentence is honest and useful.

**Quoted compiler diagnostics.** State the rule ("`warpSize` is not a constant
expression on either backend"); the transcript goes in `architecture.md`.

**Changelog.** `gpuStream_t is no longer defined here: it moved to…` — git
knows. A reader never told it used to be here does not need to be told it left.
Write the present tense.

**Self-referential repo facts.** `The fourth of the gpu* layer's device headers` —
wrong when a fifth lands, and says nothing about the file.

**Pedagogy and rhetoric.** Rhetorical section headings (`Why a third switch
point, when gpu_backend.h and runtime.cuh exist`), staged reveals (`It is
tempting to stop at namespace cg = ... and be done. That is not done here, for
two reasons`), dramatization (`which is the one thing this directory exists to
prevent`). State the rule; trust the reader.

**Second-person exhortation.** `Never treat a gpurandState* and a
gpurandStateXORWOW* as interchangeable`. Prefer the declarative: "On HIP these
are distinct types; on CUDA they are one." Where the rule is load-bearing a
`static_assert` outranks any amount of prose — `runtime.cuh`'s
power-of-two check on `WWR_WARP_SIZE` is the right pattern.

**Prose inside a section banner.** A bare `// ===` / `// Types` / `// ===`
divider is *fine* — in a 600-line list of `WWR_FUNCTION` entries it is
navigation. What is banned is the paragraph some of them wrapped. Title plus at
most one line.

**Hanging-indent prose tables.** `runtime.cuh`'s old `Provides:` block
wrapped paragraphs into a 36-column gutter: it broke on every rename and no
tool rendered it. Use a real Markdown table, or a plain two-column list.

**The same rationale in more than one place.** One canonical home; everywhere
else, a link. The `warpSize` argument was written out in `CMakeLists.txt`,
`runtime.cuh` *and* `README.md` — three copies to keep in sync.

## Put a load-bearing fact where someone will trip over it

The single most valuable move in the whole conversion. Fourteen `src/hip`
modules open with `#include <array>` (or `<algorithm>`, or `<cstdio>`) before
the vendor header, and every one is load-bearing — remove it and the build
fails on macro poisoning or an undeclared `printf`. **Not one had a comment at
the include site.** The explanation sat in a file-header paragraph, which is
the wrong altitude for a fact whose job is to stop someone deleting a line.

Each now carries two lines where the include is, with the account in
`architecture.md`. Ask of any constraint: *where is the reader standing when
they are about to get this wrong?* Put it there.

## The rewrite recipe

For each paragraph in an over-budget header, ask: **is this a constraint on the
caller, or is it the story of how we found out?** Keep the constraint, one
line. Move the story.

Before — `runtime.cuh`, ~40 lines on `WWR_WARP_SIZE` alone:

```
 *   WWR_WARP_SIZE                     the warp/wavefront size, as an integer
 *   [...] which is worth recording so it is not rediscovered:
 *     - `warpSize` is NOT a constant expression [...] nvcc 13.0: "the value
 *       of variable warpSize ... cannot be used as a constant". clang 20
 *       under HIP: "non-constexpr function 'operator int' cannot be [...]
 *     - HIP's `__AMDGCN_WAVEFRONT_SIZE__` [...] DIVERGES across HIP's two
 *       passes [...] 64 in the host pass and 32 in the device pass [...]
```

After — 2 lines:

```cpp
 *   WWR_WARP_SIZE       warp/wavefront size, as a constant expression. Set
 *                       with -DWWR_WARP_SIZE (default 32; 64 for CDNA)
```

…plus one `See docs/architecture.md (warp size)` at the foot of the header. The
nvcc quote, the host/device divergence and the ROCm version live in that section
— where version numbers belong, because `architecture.md` is dated and a header
is not.

## The long-form home

Two kinds of rationale outgrow a header, and both live in
**`docs/architecture.md`**:

- **A choice this project made**, and could revisit. "Warp size is a
  configure-time constant." "`cooperative_groups.cuh` wraps nothing." Give it
  Context and Consequences, so the decision and the alternative rejected are on
  record.
- **Something simply true of CUDA, HIP or C++** that this project must live
  with. "`hiprand_kernel.h` calls bare `printf`." "A `const` member of class
  type breaks trivial copyability under clang." Just the fact and how it was
  measured.

The test is one question: **did we decide this, or did we discover it?** — but
it now sizes the section, not which file it lands in. A discovery is a dated
fact; a decision additionally owes its Context and Consequences. Sections are
numbered, appended never inserted, since code cites them by number.

`architecture.md` opens with the toolchain every claim was measured on, and a
date, so staleness is visible:

```markdown
**Toolchain for every claim below unless a section says otherwise:** CUDA 13.0
(nvcc), clang 20.1.8 with libc++, ROCm 7.2.4, `gfx1200`, `sm_86`.
Dated 2026-09-23.
```

Cite `architecture.md` by section number.

## Declaration comments

Doxygen tags, not narrative. `@brief` on one line; `@param`, `@tparam`,
`@return` only where the name does not already say it; `@pre` for a genuine
precondition. `cooperative_groups.cuh`'s per-function blocks are the model:

```cpp
/// @brief One bit per group member, set where that member's @p pred was true
///
/// Bit i is the member of rank i. Requires a group with warp-level
/// collectives (a tile or a coalesced group), not a block or a grid.
```

Do not restate the signature in prose, and do not repeat a declaration comment
at the definition.

## CMake files

`CMakeLists.txt` and the macros in `cmake/` are code, and the same rules apply
with two mechanical differences: comments are `#`, and there is no Doxygen. A
macro or function gets a **declaration comment** — its name, what it takes, what
it guarantees — in the same 5-line spirit as a C++ `///` block; a file gets a
leading `#` header held to the same 25-line prose budget:

```cmake
# <name> — <one line>
#
# <1-3 sentences: what this file/macro provides and who uses it.>
# <Optional: constraints, one line each. Link out for the why.>
```

What breaks the budget here is identical to what broke it in the C++ headers,
and the worst offenders show every banned construct at once:

- `cmake/wwr_add_gpu_device_library.cmake` — 67% comment, opening with
  rhetorical banners (`# What this replaced`, `# What this deliberately does NOT
  take`), a staged reveal (`there is a trap in the obvious fix`), and a
  verification transcript (`verified by compiling a kernel ... under nvcc
  -arch=sm_86 ... clang -x hip --offload-arch=gfx1200 on ROCm 7.2.4`).
- `src/CMakeLists.txt` — a 77-line per-target essay that re-derives the warp-size
  host/device divergence, the *same* argument already in `runtime.cuh` and named
  as the canonical duplication under "Banned constructs" above.

The rewrite recipe is unchanged: keep the constraint one line at the command it
governs (`# PRIVATE: nothing linking this inherits -x hip`), and move the story —
the rejected alternative, the transcript, the vendor divergence — to the owning
directory's `README.md` or `docs/architecture.md`. A `# ===` divider
titling a block of commands is navigation and stays; the paragraph under it does
not.

## Checking

Budget, excluding `Usage:` blocks and tables:

```python
# scratch script — measures the leading comment block as prose
from pathlib import Path

CXX = {".cppm", ".cuh", ".h", ".cpp", ".cu"}

def cxx_header(lines):  # the leading /** ... */ block
    if not lines or not lines[0].lstrip().startswith("/**"):
        return None
    end = next((i for i, l in enumerate(lines) if l.strip() == "*/"), None)
    if end is None:
        return None
    h = lines[: end + 1]
    u = next((i for i, l in enumerate(h) if l.strip() == "* Usage:"), None)
    if u is not None:
        h = h[: u - 1 if h[u - 1].strip() == "*" else u] + [h[-1]]
    h = [l for l in h if not l.strip().lstrip("*/ ").startswith("|")]
    return len(h)

def cmake_header(lines):  # the leading run of # comments
    n = 0
    for l in lines:
        s = l.strip()
        if s.startswith("#"):
            n += 1
        elif s == "":
            if n:  # a blank ends the header once it has started
                break
        else:
            break
    return n or None

roots = [Path("src"), Path("cmake"), Path("CMakeLists.txt")]
paths = [p for r in roots for p in ([r] if r.is_file() else r.rglob("*"))]
for path in sorted(paths):
    if path.suffix in CXX:
        header = cxx_header
    elif path.name == "CMakeLists.txt" or path.suffix == ".cmake":
        header = cmake_header
    else:
        continue
    n = header(path.read_text().splitlines())
    if n is not None and n > 25:
        print(f"OVER {n:4d}  {path}")
```

Banned phrasing, which should return nothing:

```bash
grep -rn "verified by\|Verified by\|not assumed\|confirmed by\|rediscovered" \
  src/ cmake/ CMakeLists.txt \
  --include=*.cppm --include=*.cuh --include=*.h --include=*.cpp --include=*.cu \
  --include=CMakeLists.txt --include=*.cmake
```

Neither is wired into a hook or CI, deliberately: the judgement ("is this a
constraint or a story?") is not mechanizable. The checks find candidates; a
human decides.

## Scope

`src/` and `cmake/`: the C++ units (`.cppm`, `.cuh`, `.h`, `.cpp`, `.cu`), the
`CMakeLists.txt` and `*.cmake` files, and the `README.md` files under them. Not
`.claude/skills/`, `CLAUDE.md`, or `devtools/` — those are how-to documents for
people, where prose is the medium. **Shell scripts stay out** for the same
reason: every `.sh` in this repo lives in `devtools/`, `docker/`, `.devcontainer/`
or `.claude/hooks/`, all how-to-for-people territory, so a `.sh` comment budget
would reverse a decision this convention already made.

A directory `README.md` is a *destination* for displaced explanation, so it is
held to the duplication rule but not the line budget. `src/hip/README.md` is
707 lines and that is fine; what is not fine is a header restating it.

**Converting a file is a side effect of editing it, not a task of its own.** The
C++ header sweep already happened; the CMake files have not been swept, so expect
to convert one when you touch it. Move displaced material to `architecture.md` or
a README rather than deleting it, and keep each conversion reviewable. A
deliberate CMake sweep is worth doing but is its own PR — do not fold it into an
unrelated change.

## Why these rules

- **Diátaxis** (Procida) — reference, explanation, how-to and tutorial are four
  distinct modes, and mixing them degrades all four. A file header is
  reference: austere, code-shaped, complete.
- **ADRs** (Nygard, 2011; ThoughtWorks Radar "adopt") — the
  Context/Decision/Consequences shape, and the principle that a dated record is
  where measurements belong. This repo keeps no separate ADR files; it folds
  that shape into `docs/architecture.md`, alongside the facts nobody decided.
- **Google C++ Style Guide** — file comments "should be brief"; "do not
  duplicate comments in both the .h and the .cc"; declaration comments describe
  *use*, definition comments describe *operation*.
- **CppCoreGuidelines NL.1–NL.3** — don't say in comments what the code says;
  state intent; keep comments crisp.
- **LLVM Coding Standards** — a file header is "a few lines" of purpose.
  Relevant because this is a clang/libc++ project.
- **Ousterhout, *A Philosophy of Software Design*** — a comment earns its place
  by being at a *different level of detail* than the code, and interface
  comments are a different artifact from implementation comments. The 176-line
  header failed both: same level as the code (compiler flags, exact
  diagnostics), and mixing the two kinds.
- **Go doc comments / Rust API guidelines** — both fix a *small set of named
  sections* rather than free-form essays, which the template imitates.

The thing being rejected has a name: **literate programming** (Knuth) — source
as a narrative meant to be read end to end. A coherent tradition, and not what
a header serving a caller in a hurry should be.
