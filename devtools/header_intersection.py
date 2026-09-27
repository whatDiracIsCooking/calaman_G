#!/usr/bin/env python3
"""Intersect a CUDA vendor header with its HIP counterpart to find the shared API.

A ``src/<m>.cppm`` gpu* module is only as complete as the intersection of the two
vendor APIs it bridges: every symbol that BOTH cuRAND and hipRAND expose is a
symbol ``wwr.rand`` could carry a ``gpu*`` name for, and any it skips is a
coverage hole. This script computes that intersection straight from the vendor
``.h`` files, so "did we cover everything the two backends agree on?" becomes a
diff instead of a manual read of two headers.

It is the OUTWARD-facing complement to ``test/shared/alias_coverage.py``: that
one asks "is every alias we defined independently verified?", this one asks "is
every symbol the two backends share actually aliased?". Neither implies the
other.

WHAT COUNTS AS A SYMBOL. The public surface of these libraries is exactly its
vendor-prefixed identifiers -- ``curand*`` / ``CURAND_*`` on one side,
``hiprand*`` / ``HIPRAND_*`` on the other -- covering functions, typedefs, enum
tags, enum constants and macros alike. So rather than parse C (the vendor
headers pull in device intrinsics that no host parser digests cleanly), we take
every identifier whose leading prefix matches the library's, strip that prefix,
and compare the remainder case-insensitively:

    curandCreateGenerator -> "creategenerator" <- hiprandCreateGenerator   (match)
    CURAND_STATUS_SUCCESS  -> "status_success"  <- HIPRAND_STATUS_SUCCESS   (match)

The prefix is auto-detected as the most common leading token on each side
(``curand`` beats the stray ``cudaStream_t`` cross-reference), and can be pinned
with ``--cuda-prefix`` / ``--hip-prefix`` for the odd library. Comments and the
include guard are dropped so prose and ``#ifndef CURAND_H_`` do not masquerade as
API.

WHAT THE OUTPUT PROVES, AND WHAT IT DOESN'T. The INTERSECTION is the reliable
list: a symbol only lands there when both headers spell it, so internal noise
(``CURAND_KNUTH`` and friends, method-enum guts with no HIP twin) self-filters.
The CUDA-only / HIP-only lists explain the "absent for want of a counterpart"
notes a module header carries -- but they are noisier, since a backend's private
identifiers show up there too. And matching is by name: it confirms a shared
NAME exists, not that the two share a signature or an enum VALUE (see rand.cppm
on why the values differ). Read a green ``--coverage`` run as "every shared name
is aliased", not "the aliases are correct" -- that is what dispatch.py and the
compiler are for.

``--coverage`` matches by LITERAL name, so read its "missing" list against the
module's CONTRACT, which comes in two kinds.

A WHOLE-SURFACE module (rand, fft, tx) promises to wrap everything the two
backends share and spells both names out (``WWR_FUNCTION(gpu, cu, hip)``).
For these ``--coverage`` is the real completeness gate: a nonempty "missing" is a
genuine hole (or a documented omission the module names in its header).

A CURATED-SUBSET module (blas, solver, sparse, runtime_api) lists only the names
the layer above it uses ("Only the names src/wrappers/blas uses are listed") --
it never promised the full intersection, so its "missing" list is
reachable-but-unused vendor symbols, a discovery menu, not a defect report.
blas.cppm still spells its names out in full, so ``--coverage`` covers most of
the intersection (~300/320) and the tail is the unused remainder; solver and
sparse read as near-empty for a different reason -- cusolverDn* vs hipsolver* and
cusparse's opaque generic API barely intersect by NAME at all, so the scan is
blind there. For all four, completeness is enforced elsewhere: the compiler (an
unresolved gpu* name cannot be consumed), ``test/shared/alias_coverage.py``
(every alias defined has a test), and the dispatch tables that
``test/shared/dispatch.py`` checks (every wrapper calls the right alias).

Pass a module's ``.cuh`` alongside its ``.cppm`` when device-side names live there.

Usage:
    devtools/header_intersection.py \\
        --cuda /usr/local/cuda/include/curand.h \\
               /usr/local/cuda/include/curand_kernel.h \\
        --hip  /opt/rocm/include/hiprand/hiprand.h \\
               /opt/rocm/include/hiprand/hiprand_kernel.h

    # Confirm the gpu* rand layer wraps the whole shared surface:
    devtools/header_intersection.py --cuda curand.h --hip hiprand.h \\
        --coverage src/rand.cppm src/rand.cuh
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
# An include guard: an all-caps identifier ending in _H or _H_ (CURAND_H_,
# HIPRAND_H_). Left in, its "_H_" remainders would falsely intersect.
_GUARD_RE = re.compile(r"_H_?$")


def strip_comments(text: str) -> str:
    """Drop // and /* */ comments so identifiers named only in prose don't count.

    A small state machine rather than a regex: it must not treat // inside a
    /* */ block, or a /* opened inside a // line, as a real delimiter.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        two = text[i : i + 2]
        if two == "//":
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif two == "/*":
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def leading_prefix(ident: str) -> str | None:
    """The library token an identifier leads with, lowercased.

    ``curandFoo`` and ``cudaMalloc`` lead with a lowercase run (``curand`` /
    ``cuda``); ``CURAND_FOO`` leads with an all-caps run up to the first
    separator (``CURAND`` -> ``curand``). Both forms of one library normalise to
    the same token, which is what lets the two spellings vote together.
    """
    if m := re.match(r"[a-z]+", ident):
        return m.group(0)
    if m := re.match(r"[A-Z]+", ident):
        return m.group(0).lower()
    return None


def detect_prefix(idents: set[str], family: str) -> str | None:
    """The dominant library prefix on one side (mode of the leading tokens).

    ``family`` seeds which side we're on -- ``cu`` for CUDA, ``hip`` for HIP --
    so a curand header's stray ``cudaStream_t`` still counts toward a ``cu*``
    prefix but ``size_t`` does not, and the mode picks ``curand`` over the lone
    ``cuda``.
    """
    counts: Counter[str] = Counter()
    for ident in idents:
        p = leading_prefix(ident)
        if p and p.startswith(family):
            counts[p] += 1
    return counts.most_common(1)[0][0] if counts else None


def collect_identifiers(paths: list[Path]) -> tuple[set[str], set[str]]:
    """Union of identifiers across the given headers, and which are called.

    Returns ``(all_identifiers, called_identifiers)`` -- the second is every name
    that appears as ``name(`` somewhere, used only to label a symbol ``func``.
    """
    idents: set[str] = set()
    called: set[str] = set()
    for path in paths:
        text = strip_comments(path.read_text(errors="replace"))
        idents.update(_IDENT_RE.findall(text))
        called.update(_CALL_RE.findall(text))
    return idents, called


def classify(ident: str, called: set[str]) -> str:
    """Best-effort kind tag: func, const, or type."""
    if ident in called:
        return "func"
    if ident.isupper() or re.fullmatch(r"[A-Z0-9_]+", ident):
        return "const"
    return "type"


class Side:
    """One backend's API surface: prefix, and normalised-key -> names it maps to."""

    def __init__(self, name: str, paths: list[Path], family: str, prefix: str | None):
        self.name = name
        self.paths = paths
        idents, self.called = collect_identifiers(paths)
        self.prefix = prefix or detect_prefix(idents, family)
        # normalised key -> the actual identifier(s) that produced it
        self.by_key: dict[str, set[str]] = {}
        if not self.prefix:
            return
        plen = len(self.prefix)
        for ident in idents:
            if leading_prefix(ident) != self.prefix:
                continue
            if _GUARD_RE.search(ident) and ident.isupper():
                continue
            key = ident[plen:].lstrip("_").lower()
            if key:
                self.by_key.setdefault(key, set()).add(ident)
        self._fold_tags()

    def _fold_tags(self) -> None:
        """Fold a C struct/enum tag into the public typedef that stands for it.

        ``curandGenerator_st`` (struct tag) and ``curandOrdering`` (enum tag) are
        the implementation spellings of ``curandGenerator_t`` / ``curandOrdering_t``
        -- the ``_t`` is the name the gpu* layer aliases, and wrapping the raw tag
        would be pointless. Folding the tag's names into its ``_t`` sibling keeps
        it from being reported as an uncovered "shared symbol". Applied identically
        to both backends, so it never drops a name the two genuinely share.
        """
        for key in list(self.by_key):
            if key.endswith("_t"):
                continue
            base = key[:-3] if key.endswith("_st") else key
            canon = base + "_t"
            if canon in self.by_key:
                self.by_key[canon] |= self.by_key.pop(key)

    def names(self, key: str) -> list[str]:
        return sorted(self.by_key.get(key, ()))

    def kind(self, key: str) -> str:
        return classify(sorted(self.by_key[key])[0], self.called)


def tokens(text: str) -> set[str]:
    return set(_IDENT_RE.findall(text))


def build_report(cuda: Side, hip: Side, coverage: list[Path] | None) -> dict:
    """The full comparison as plain data, ready for text or JSON rendering."""
    ckeys, hkeys = set(cuda.by_key), set(hip.by_key)

    def entry(side: Side, key: str) -> dict:
        return {"names": side.names(key), "kind": side.kind(key)}

    intersection = [
        {"key": k, "cuda": entry(cuda, k), "hip": entry(hip, k)}
        for k in sorted(ckeys & hkeys)
    ]
    report = {
        "cuda": {
            "prefix": cuda.prefix,
            "sources": [str(p) for p in cuda.paths],
            "symbols": len(cuda.by_key),
        },
        "hip": {
            "prefix": hip.prefix,
            "sources": [str(p) for p in hip.paths],
            "symbols": len(hip.by_key),
        },
        "intersection": intersection,
        "cuda_only": [
            {"key": k, **entry(cuda, k)} for k in sorted(ckeys - hkeys)
        ],
        "hip_only": [
            {"key": k, **entry(hip, k)} for k in sorted(hkeys - ckeys)
        ],
    }

    if coverage:
        present: set[str] = set()
        for path in coverage:
            present |= tokens(path.read_text())
        missing = [
            item
            for item in intersection
            if not (present & set(item["cuda"]["names"]))
            and not (present & set(item["hip"]["names"]))
        ]
        report["coverage"] = {
            "module": ", ".join(str(p) for p in coverage),
            "intersection": len(intersection),
            "missing": missing,
        }
    return report


def _fmt(side_entry: dict) -> str:
    return "|".join(side_entry["names"])


def render_text(report: dict, show: str) -> str:
    out: list[str] = []
    c, h = report["cuda"], report["hip"]
    out.append(f"CUDA  prefix={c['prefix']!r}  {c['symbols']} symbols  "
               f"({', '.join(c['sources'])})")
    out.append(f"HIP   prefix={h['prefix']!r}  {h['symbols']} symbols  "
               f"({', '.join(h['sources'])})")
    out.append("")

    inter = report["intersection"]
    if show in ("all", "intersection"):
        out.append(f"== Intersection: {len(inter)} shared symbol(s) ==")
        width = max((len(_fmt(i["cuda"])) for i in inter), default=0)
        for item in inter:
            out.append(f"  [{item['cuda']['kind']:5}] "
                       f"{_fmt(item['cuda']):{width}}  {_fmt(item['hip'])}")
        out.append("")

    if show in ("all", "cuda-only"):
        only = report["cuda_only"]
        out.append(f"== CUDA-only: {len(only)} symbol(s) with no HIP counterpart ==")
        for item in only:
            out.append(f"  [{item['kind']:5}] {'|'.join(item['names'])}")
        out.append("")

    if show in ("all", "hip-only"):
        only = report["hip_only"]
        out.append(f"== HIP-only: {len(only)} symbol(s) with no CUDA counterpart ==")
        for item in only:
            out.append(f"  [{item['kind']:5}] {'|'.join(item['names'])}")
        out.append("")

    if "coverage" in report:
        cov = report["coverage"]
        missing = cov["missing"]
        covered = cov["intersection"] - len(missing)
        out.append(f"== Coverage in {cov['module']} ==")
        out.append(f"  {cov['intersection']} shared symbols, "
                   f"{covered} referenced, {len(missing)} MISSING")
        for item in missing:
            out.append(f"    MISSING [{item['cuda']['kind']:5}] "
                       f"{_fmt(item['cuda'])}  {_fmt(item['hip'])}")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cuda", nargs="+", type=Path, required=True,
                        metavar="HEADER", help="CUDA vendor header(s)")
    parser.add_argument("--hip", nargs="+", type=Path, required=True,
                        metavar="HEADER", help="HIP vendor header(s)")
    parser.add_argument("--cuda-prefix", help="pin the CUDA prefix (else auto)")
    parser.add_argument("--hip-prefix", help="pin the HIP prefix (else auto)")
    parser.add_argument("--coverage", nargs="+", type=Path, metavar="SRC",
                        help="src file(s) -- e.g. a gpu* module and its .cuh -- to "
                             "check the intersection against by literal name; a "
                             "shared symbol counts as covered when either backend's "
                             "name appears verbatim (so token-pasted dispatch call "
                             "sites, which name no vendor symbol, won't match)")
    parser.add_argument("--show", choices=["all", "intersection",
                                           "cuda-only", "hip-only"],
                        default="intersection", help="which sections to print")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(argv)

    cuda = Side("cuda", args.cuda, "cu", args.cuda_prefix)
    hip = Side("hip", args.hip, "hip", args.hip_prefix)
    for side in (cuda, hip):
        if not side.prefix:
            print(f"header_intersection: no {side.name} vendor prefix detected in "
                  f"{', '.join(map(str, side.paths))}; pass --{side.name}-prefix",
                  file=sys.stderr)
            return 2

    report = build_report(cuda, hip, args.coverage)

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report, args.show), end="")

    # Exit 1 when a coverage gap was found, so this can gate in a script.
    if "coverage" in report and report["coverage"]["missing"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
