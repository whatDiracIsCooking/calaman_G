---
name: workspace
description: >-
  The carve-once device-workspace convention — one caller-provided buffer, one
  carve() per routine, sized and carved by the same code path through
  calaman::carve_workspace / WorkspaceLayout. Use when writing a routine that
  needs device scratch, adding or reviewing a *_bufferSize entry point, carving
  a buffer into regions, or when workspace sizing and carving look like they
  could disagree. Not for documentation style (that's `docstyle`) or for
  running the suites (that's `test`).
---

# Device workspaces: size once, carve once

Every routine here that needs device memory takes **one caller-provided
buffer** and carves it into aligned regions — the shipped surface allocates
nothing (CLAUDE.md). The invariant this skill protects: **the byte count a
`*_bufferSize` reports and the pointers the routine carves must come from the
same code path**, so they cannot drift. The machinery lives in
`calaman.common`'s `:workspace` partition (`src/common/workspace.cppm`); the
host-only suite `test/common/workspace_tests.cpp` pins its arithmetic.

Two names, outermost first — prefer the outermost that fits:

| Name | Reach for it when |
|---|---|
| `carve_workspace` + the `slices_for` concept | the routine hands out two or more region pointers — the normal case |
| `WorkspaceLayout` | writing a `carve()` member (it's the parameter type), or a sizing-only query with no pointers to hand out (`horner_bufferSize`) |

`WorkspaceBuilder` no longer exists (#139). If you find yourself writing a
sizing list in one place and a pointer-carving loop in another, you are
reinventing it — that separately-maintained mirror is exactly the drift the
convention deleted (it had produced near-misses in cg_unitary and nnls before
#137 removed both).

## The canonical shape

One struct, one `carve()`, run twice. `XSlices` (or `XWorkspace` for a
module-internal one) owns every region pointer and a
`carve(WorkspaceLayout &, shape...)` member that is the ONLY description of
the layout:

```cpp
template<wwr::real_fp T>
struct XSlices {
  T *a = nullptr;          // every member default-initialized: slices_for
  int *info = nullptr;     // requires std::default_initializable
  T *eig_scratch = nullptr;
  int lwork_eig = 0;

  void carve(WorkspaceLayout &layout, const int n, const int eig_len) {
    lwork_eig = eig_len;
    a = layout.fixed<T>(static_cast<std::size_t>(n) * n);
    info = layout.fixed<int>(1);
    eig_scratch = layout.scratch<T>(static_cast<std::size_t>(eig_len));
  }
};
```

Around it, two entry points, split by what they know:

- **`make_X_slices` / `map_workspace`** — validation plus the vendor lwork
  queries (`syevd_bufferSize`, `orgqr_bufferSize`, …), then ONE
  `carve_workspace(d_work, out, shape...)` call. The queries live here, never
  inside `carve()` — carve is pure layout, callable with no handle.
- **`X_bufferSize`** — forwards with nulls:
  `carve_workspace<XSlices<T>>(nullptr, nullptr, shape...)` sizes without
  carving (or forwards to `make_X_slices` with null slices, davidson-style).

Inside the routine, one call serves both the undersized-buffer check and the
pointers:

```cpp
XSlices<T> s;
if (work_bytes < carve_workspace(d_work, &s, n, eig_len)) {
  return wwr::WWRBLAS_STATUS_INVALID_VALUE; // or ALLOC_FAILED, per module
}
```

Where it lives: a dedicated `:buffer_size` partition when the module has one
(davidson, feast, expm, cg_unitary), inline in the module's interface for the
small `src/lapack/` modules (geev, orghr) and nnls.

## The rules the layout depends on

- **Every `fixed()` before any `scratch()`.** Fixed regions accumulate;
  scratch regions all alias ONE zone at the end of the fixed run and
  `total()` counts only the largest. A `fixed()` after a `scratch()` would
  overlap the zone. (Enforced by nothing — this is the one ordering mistake
  the machinery cannot catch.)
- **Scratch is for never-live-together temporaries.** Two solver workspaces
  used in different phases share one block sized to the larger: davidson's
  syevd/sygvd, feast's QR/eigensolver, cg_unitary's expm/cost-functor pair.
  When the caller already folded the max into one length (davidson's
  `lwork_eig`), one `scratch()` call carries it.
- **Null-base mode must be safe.** Over a null base every `fixed()`/
  `scratch()` returns nullptr while advancing offsets identically. Derived
  pointers — `ints + 1`, `base + offsetof(...)` — need a null guard or they
  manufacture non-null garbage in sizing mode (see nnls's int block and
  feast's `FeastStatus` field pointers).
- **Conditional regions are fine** — carve them under the same flag the
  routine branches on, so the unused path costs nothing (davidson's
  `with_metric` block, geev's `wantvl`/`wantvr` matrices).
- **256-byte alignment is the default everywhere.** Regions start aligned
  because device allocators hand back 256-aligned bases; when a stride must
  be whole elements, divide the aligned byte count by `sizeof` (feast's
  resolvent stride). `align_up` (`:align_up`) is the shared rounding.
- **Name carve parameters apart from the fields they set** (`eig_len` param,
  `lwork_eig` field) — a parameter shadowing the member it assigns is the
  bug waiting in every member-carve.
- **No overflow hardening**, deliberately: plain `std::size_t` arithmetic for
  realistic sizes, not adversarial inputs (`workspace.cppm`, `align_up.h`).

## Beyond the canonical shape

- **Composed layouts** (expm): a struct reserves its own regions and exposes
  the next free offset via `layout.cursor()` as a sub-layout base
  (`ExpmWorkspace::pade_base`); the driver sizes the tail separately —
  `expm_bufferSize` maxes the `PadeWorkspace` carve over every Padé degree
  and adds it to the `ExpmWorkspace` span, because the degree is chosen at
  run time.
- **A shared scratch zone's base** is itself a zero-length scratch call:
  `scratch<std::byte>(0)` after the per-user scratch declarations returns the
  zone's base without widening it (geev).
- **Sizing-only, one region**: skip the struct — `WorkspaceLayout
  layout(nullptr); (void)layout.scratch<T>(count); return layout.total();`
  (horner).

## Reviewing workspace code

Ask, in order:

1. Is there exactly ONE description of the layout — a `carve()` the sizing
   and the carving both run? Any hand bump-pointer arithmetic or a second
   region list is the drift bug, whatever it's called.
2. Do all `fixed()` calls precede all `scratch()` calls?
3. Is null-base mode safe (no derived-pointer arithmetic without a guard)?
4. Do the vendor lwork queries sit outside `carve()`?
5. Does the struct satisfy `slices_for` (default-initializable, carve member)?
   `static_assert(slices_for<XSlices<T>, ...>)` is cheap insurance, and the
   concept's name is the convention's greppable marker.

Worked examples, simplest first: `src/horner/interface.cppm` (sizing-only),
`src/lapack/orghr/interface.cppm` (two regions), `src/nnls/nnls.cppm`
(module-internal struct, derived pointers), `src/davidson/buffer_size.cppm`
(conditional regions + shared scratch), `src/feast/buffer_size.cppm` (strides,
`offsetof` fields), `src/lapack/geev/geev.cppm` (conditional + five scratch
users + zero-length zone base), `src/expm/detail.cppm` + `buffer_size.cppm`
(composed sub-layout). History: #137 (machinery + the two drift-prone
migrations), #138 (adoption), #139 (WorkspaceBuilder deleted).
