#!/usr/bin/env bash
# Print the transitive call graph of a reference-LAPACK routine, read from its
# Fortran source, and say which of its LAPACK callees calaman already ships.
#
#   devtools/lapack-callgraph.sh DGELQ          edges, then the calaman gap
#   devtools/lapack-callgraph.sh --edges ZGEQP3 edges only
#   devtools/lapack-callgraph.sh --refresh DGELQ  re-download cached sources
#   devtools/lapack-callgraph.sh --netlib DGELQ   ... and diff against netlib
#
# --netlib checks every routine reached against the release tarball netlib
# serves (www.netlib.org/lapack/lapack.tgz, the source its explore-html pages
# are generated from) and appends one line per discrepancy: a routine absent
# from one side, callees only one side has, or executable code that differs
# (comments and whitespace ignored; the counts are statements present only on
# each side, and a routine whose sole change is the IMPLICIT NONE that master
# added tree-wide after 3.12.1 says so). It names the tarball's version, since
# netlib, LAPACK_REF and the distro LAPACK the tests use can be three releases.
#
# One `ROUTINE -> CALLEE ...` line per routine reached, breadth-first from the
# root. A callee is anything named by a CALL or an EXTERNAL declaration in the
# routine's executable source (comment lines stripped, fixed-form continuation
# lines joined), so functions such as ILAENV and DLAMCH count too.
#
# Sources come from Reference-LAPACK/lapack on GitHub (LAPACK_REF, default
# master), tried in SRC, BLAS/SRC and INSTALL, as .f then .f90, and cached in
# LAPACK_SRC_CACHE (default ~/.cache/calaman/lapack-src/<ref>). A routine found
# in none of them is reported MISSING rather than failing the walk.
#
# Exit status: 0 when the walk finished, 1 on a usage error or when the root
# itself cannot be fetched.
#:
# The calaman gap maps a SRC routine to src/lapack/<name minus its s/d/c/z
# prefix>/ -- so DLARFG and ZLARFG both resolve to larfg/, which says the
# module exists, not that it instantiates that precision. BLAS and INSTALL
# routines, the ILAENV/ILAPREC/ILAUPLO-style enum helpers, XERBLA, LSAME and
# ?ISNAN are left out of the gap: they belong to the vendor BLAS and to
# calaman.common, not to a routine module.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

usage() { usage_from_header "${BASH_SOURCE[0]}"; }

REF=${LAPACK_REF:-master}
BASE=https://raw.githubusercontent.com/Reference-LAPACK/lapack/$REF
CACHE_ROOT=${XDG_CACHE_HOME:-$HOME/.cache}/calaman/lapack-src
CACHE_ROOT=${LAPACK_SRC_CACHE:-$CACHE_ROOT}
CACHE=$CACHE_ROOT/$REF
NETLIB_TGZ=https://www.netlib.org/lapack/lapack.tgz

edges_only=0
refresh=0
netlib=0
root=
while [ $# -gt 0 ]; do
  case $1 in
    --edges) edges_only=1 ;;
    --refresh) refresh=1 ;;
    --netlib) netlib=1 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
    *) [ -z "$root" ] || { usage >&2; exit 1; }; root=$1 ;;
  esac
  shift
done
[ -n "$root" ] || { usage >&2; exit 1; }
root=$(echo "$root" | tr "[:lower:]" "[:upper:]")

mkdir -p "$CACHE"
if [ "$refresh" -eq 1 ]; then
  rm -f "$CACHE"/*.f "$CACHE"/*.f90 "$CACHE"/*.where
  rm -rf "$CACHE_ROOT/netlib"
fi

# The source file name for routine $1: lower case, and ?LAMC3, which has no
# file of its own, mapped to the ?lamch.f it is the second unit of.
src_name() {
  local name
  name=$(echo "$1" | tr "[:upper:]" "[:lower:]")
  case $name in [sd]lamc3) name=${name%c3}ch ;; esac
  echo "$name"
}

# The first of the paths given that exists, or nothing.
first_file() {
  local f
  for f in "$@"; do [ -f "$f" ] && { echo "$f"; return; }; done
  return 0
}

# Fetch routine $1 into the cache; print its local path, or nothing if absent.
# The directory it came from is remembered in <name>.where for the gap report.
fetch() {
  local name dir ext
  name=$(src_name "$1")
  for ext in f f90; do
    if [ -s "$CACHE/$name.$ext" ]; then echo "$CACHE/$name.$ext"; return; fi
  done
  for dir in SRC BLAS/SRC INSTALL; do
    for ext in f f90; do
      if curl -sfL --max-time 20 "$BASE/$dir/$name.$ext" \
          -o "$CACHE/$name.$ext"; then
        echo "$dir" > "$CACHE/$name.where"
        echo "$CACHE/$name.$ext"
        return
      fi
      rm -f "$CACHE/$name.$ext"
    done
  done
}

# The executable statements of source file $1, upper-cased, one per line:
# comments dropped and fixed-form continuation lines joined.
code() {
  awk '
    # fixed-form comment lines (.f): C, c, * or ! in column 1
    FILENAME ~ /\.f$/ && /^[Cc*!]/ { next }
    { sub(/!.*/, "") }
    # fixed-form continuation: non-blank, non-zero column 6
    FILENAME ~ /\.f$/ && length($0) >= 6 && substr($0, 6, 1) !~ /[ 0]/ {
      line = line " " substr($0, 7); next
    }
    { if (line != "") print line; line = $0 }
    END { if (line != "") print line }
  ' "$1" | tr "[:lower:]" "[:upper:]"
}

# Callees of source file $1: CALL targets plus EXTERNAL names, one per line.
callees() {
  code "$1" | awk '
    {
      s = $0
      while (match(s, /CALL +[A-Z0-9_]+/)) {
        t = substr(s, RSTART, RLENGTH); sub(/CALL +/, "", t); print t
        s = substr(s, RSTART + RLENGTH)
      }
    }
    /^ *EXTERNAL[ :]/ {
      sub(/^ *EXTERNAL *(:: *)?/, ""); gsub(/ /, "")
      n = split($0, names, ","); for (i = 1; i <= n; i++) print names[i]
    }
  ' | grep -v '^$' | sort -u
}

declare -A seen=()
queue=("$root")
order=()
missing=()
while [ ${#queue[@]} -gt 0 ]; do
  r=${queue[0]}
  queue=("${queue[@]:1}")
  [ -n "${seen[$r]:-}" ] && continue
  seen[$r]=1
  src=$(fetch "$r")
  if [ -z "$src" ]; then
    [ "$r" = "$root" ] && { echo "cannot fetch $r at $REF" >&2; exit 1; }
    missing+=("$r")
    continue
  fi
  order+=("$r")
  mapfile -t out < <(callees "$src")
  echo "$r -> ${out[*]:-}"
  queue+=("${out[@]}")
done

for m in "${missing[@]}"; do
  case $m in
    BLAS_*_X) echo "$m MISSING (XBLAS, not part of reference LAPACK)" ;;
    *) echo "$m MISSING" ;;
  esac
done

if [ "$netlib" -eq 1 ]; then
  ndir=$CACHE_ROOT/netlib
  if [ ! -d "$ndir/src" ]; then
    mkdir -p "$ndir"
    curl -sfL --max-time 120 "$NETLIB_TGZ" -o "$ndir/lapack.tgz" \
      || { echo "cannot fetch $NETLIB_TGZ" >&2; exit 1; }
    mkdir -p "$ndir/src"
    tar xzf "$ndir/lapack.tgz" -C "$ndir/src" --strip-components=1
    tar tzf "$ndir/lapack.tgz" | head -1 | tr -d / > "$ndir/topdir"
  fi
  ver_re='s/^set\(LAPACK_(MAJOR|MINOR|PATCH)_VERSION ([0-9]+)\)/\2/p'
  cmake_ver=$(sed -nE "$ver_re" "$ndir/src/CMakeLists.txt" | paste -sd.)
  echo
  echo "netlib $(cat "$ndir/topdir") (its CMakeLists.txt says $cmake_ver)" \
    "vs GitHub $REF:"
  netlib_file() {
    first_file "$ndir"/src/{SRC,BLAS/SRC,INSTALL}/"$1".{f,f90}
  }
  ndiff=0
  for r in "${order[@]}"; do
    name=$(src_name "$r")
    gh=$(first_file "$CACHE/$name".{f,f90})
    nl=$(netlib_file "$name")
    if [ -z "$nl" ]; then
      echo "  $r: not in netlib"; ndiff=$((ndiff + 1)); continue
    fi
    only_gh=$(comm -23 <(callees "$gh") <(callees "$nl") | paste -sd' ')
    only_nl=$(comm -13 <(callees "$gh") <(callees "$nl") | paste -sd' ')
    notes=()
    [ -n "$only_gh" ] && notes+=("calls only on GitHub: $only_gh")
    [ -n "$only_nl" ] && notes+=("calls only on netlib: $only_nl")
    d=$(diff <(code "$gh" | tr -s ' \t' ' ') <(code "$nl" | tr -s ' \t' ' ') \
      || true)
    if [ -n "$d" ]; then
      n_gh=$(grep -c '^<' <<< "$d" || true)
      n_nl=$(grep -c '^>' <<< "$d" || true)
      if [ "$n_gh" -eq 1 ] && [ "$n_nl" -eq 0 ] \
          && grep -qE '^< *IMPLICIT NONE$' <<< "$d"; then
        notes+=("only IMPLICIT NONE added")
      else
        notes+=("code differs (GitHub +$n_gh, netlib +$n_nl statements)")
      fi
    fi
    if [ ${#notes[@]} -gt 0 ]; then
      ndiff=$((ndiff + 1))
      line="  $r: ${notes[0]}"
      for n in "${notes[@]:1}"; do line="$line; $n"; done
      echo "$line"
    fi
  done
  for m in "${missing[@]}"; do
    if [ -n "$(netlib_file "$(src_name "$m")")" ]; then
      echo "  $m: only in netlib"; ndiff=$((ndiff + 1))
    fi
  done
  if [ "$ndiff" -eq 0 ]; then
    echo "  no discrepancies in the ${#order[@]} routines"
  fi
fi
[ "$edges_only" -eq 1 ] && exit 0

echo
echo "calaman modules for the LAPACK (SRC) routines reached:"
for r in "${order[@]}"; do
  name=$(echo "$r" | tr "[:upper:]" "[:lower:]")
  [ "$(cat "$CACHE/$name.where" 2>/dev/null)" = SRC ] || continue
  case $name in
    ilaenv|ieeeck|iparmq|ilaprec|ilauplo|ilatrans|xerbla|lsame)
      continue ;;  # host-side plumbing
    [sd]isnan|[sd]laisnan) continue ;;
  esac
  mod=${name#[sdcz]}
  if [ -d "$REPO_ROOT/src/lapack/$mod" ]; then
    printf '  have     %-20s src/lapack/%s\n' "$r" "$mod"
  else
    printf '  missing  %-20s calaman.%s\n' "$r" "$mod"
  fi
done
