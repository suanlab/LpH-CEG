"""Widen the Step-5 certificate: every hyperpath, block sizes 1-3.

`shannon_implication_check.py` certifies the Step-5 obligation over the
polymatroid cone for 12 shapes, but only for the ONE hyperpath the
estimator returns at bs=1 (reviewer R1, [M2]).  The obligation depends on
a hyperpath only through its step structure -- the sequence of
(attributes of the block, interface with what is already covered) -- so
it can be decided for every hyperpath a shape admits, at every block
size, by enumerating those structures.

Enumeration: a hyperpath is an ordered partition of the relations into
blocks of size <= bs where every block after the first shares at least
one attribute with the relations already placed.  Distinct orderings that
induce the same step structure are checked once.  Enumeration is
depth-first in a fixed order and capped at MAX_STRUCTURES per (shape,
bs); checking stops after TIME_BUDGET_S.  The CSV records both, so "every
hyperpath" is claimed only where the enumeration finished AND every
enumerated structure was checked (`complete` column).  The vertex
enumeration is exhaustive per structure (C(|V|+3m, |V|) candidate active
sets), which is what makes k5 at bs=1 (m=10) the expensive corner.

For each structure the check is the complete one from the base script:
every vertex of the cover polytope plus its recession rays, which decides
the obligation for all feasible weight vectors.  One reduction: once the
rays are known non-negative the obligation is monotone in w, so a vertex
that dominates another componentwise cannot be the minimiser and is
skipped (`minimal_vertices`).  If the ray check fails the pair is reported
REFUTED without the reduction.

Output: results/validity/shannon_wide.csv
"""

from __future__ import annotations

import csv
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scripts.shannon_implication_check import (
    FAMILIES,
    MAX_VERTICES,
    TOL,
    Step,
    _vec_to_w,
    check_obligation,
    cover_matrix,
    enumerate_vertices,
    rays_nonnegative,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parent.parent
BLOCK_SIZES = (1, 2, 3)
MAX_STRUCTURES = 400
TIME_BUDGET_S = 1800.0  # per (shape, bs)
WORKERS = 12
MAX_ATTRS = 6  # 2^|V| polymatroid LP per vertex; |V|=7 is left to the base script
ONLY = set(sys.argv[1:])


def structures(
    edges: list[tuple[str, str]], idx: dict[str, int], bs: int
) -> tuple[list[tuple[Step, ...]], bool]:
    """All distinct step structures of hyperpaths with blocks of size <= bs."""
    rel_masks = [(1 << idx[u]) | (1 << idx[v]) for u, v in edges]
    m = len(rel_masks)
    out: list[tuple[Step, ...]] = []
    seen: set[tuple[Step, ...]] = set()
    exhausted = True

    def dfs(used: int, covered: int, seq: tuple[Step, ...]) -> None:
        nonlocal exhausted
        if len(out) >= MAX_STRUCTURES:
            exhausted = False
            return
        if used == (1 << m) - 1:
            if seq not in seen:
                seen.add(seq)
                out.append(seq)
            return
        free = [i for i in range(m) if not used & (1 << i)]
        # blocks: subsets of free relations of size <= bs, connected to `covered`
        for size in range(1, bs + 1):
            for combo in _combos(free, size):
                a = 0
                for i in combo:
                    a |= rel_masks[i]
                iface = a & covered
                if covered and not iface:
                    continue
                st = Step(block_mask=a, iface_mask=iface)
                u = used
                for i in combo:
                    u |= 1 << i
                dfs(u, covered | a, (*seq, st))

    dfs(0, 0, ())
    return out, exhausted


def minimal_vertices(verts: list[np.ndarray]) -> list[np.ndarray]:
    """Drop vertices that dominate another vertex componentwise."""
    if not verts:
        return verts
    arr = np.array(verts)
    keep: list[np.ndarray] = []
    for i, v in enumerate(arr):
        dominated_by_other = np.all(arr <= v + 1e-9, axis=1)
        dominated_by_other[i] = False
        if not dominated_by_other.any():
            keep.append(v)
    return keep


def _combos(items: list[int], size: int) -> Iterator[tuple[int, ...]]:
    return itertools.combinations(items, size)


def check_pair(fam: str, bs: int) -> dict[str, object]:
    """Run the complete check for one (shape, block size) pair."""
    edges = FAMILIES[fam]
    attrs = sorted({a for e in edges for a in e})
    n = len(attrs)
    idx = {a: i for i, a in enumerate(attrs)}
    t0 = time.perf_counter()
    structs, exhausted = structures(edges, idx, bs)
    worst = float("inf")
    all_ok = True
    n_vertices = 0
    n_checked = 0
    for seq in structs:
        if time.perf_counter() - t0 > TIME_BUDGET_S:
            break
        n_checked += 1
        steps = list(seq)
        cov = cover_matrix(steps, attrs, idx)
        verts = enumerate_vertices(cov, MAX_VERTICES)
        n_vertices += len(verts)
        ok_rays = rays_nonnegative(steps, n)
        probe = minimal_vertices(verts) if ok_rays else verts
        w_slack = min(
            (check_obligation(steps, _vec_to_w(v), n)[0] for v in probe),
            default=0.0,
        )
        worst = min(worst, w_slack)
        if w_slack < -TOL or not ok_rays or len(verts) >= MAX_VERTICES:
            all_ok = False
    dt = time.perf_counter() - t0
    complete = exhausted and n_checked == len(structs)
    return {
        "family": fam,
        "n_attrs": n,
        "block_size": bs,
        "n_structures_enumerated": len(structs),
        "n_structures_checked": n_checked,
        "enumeration_exhausted": exhausted,
        "complete": complete,
        "n_vertices_total": n_vertices,
        "worst_slack": f"{worst:.6f}",
        "all_hold": all_ok,
        "seconds": f"{dt:.1f}",
    }


def main() -> None:
    print("Step-5 obligation over ALL hyperpaths, bs in {1,2,3} (polymatroid cone)")
    print("=" * 84)
    print(
        f"  {'family':<10} {'|V|':>3} {'bs':>2} {'checked/enum':>11}"
        f"{'complete':>9} {'worst slack':>12}  verdict   time"
    )
    print(f"  {'-' * 80}")
    pairs = [
        (fam, bs)
        for fam, edges in FAMILIES.items()
        if (not ONLY or fam in ONLY) and len({a for e in edges for a in e}) <= MAX_ATTRS
        for bs in BLOCK_SIZES
    ]
    rows_out: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(len(pairs), WORKERS)) as pool:
        for r in pool.map(check_pair, *zip(*pairs, strict=True)):
            rows_out.append(r)
            print(
                f"  {r['family']:<10} {r['n_attrs']:>3} {r['block_size']:>2} "
                f"{r['n_structures_checked']:>5}/{r['n_structures_enumerated']:<5}"
                f"{r['complete']!s:>9} {r['worst_slack']:>12}  "
                f"{'HOLDS' if r['all_hold'] else 'REFUTED'}   {r['seconds']:>6}s",
                flush=True,
            )
    n_ok = sum(1 for r in rows_out if r["all_hold"])
    n_complete = sum(1 for r in rows_out if r["all_hold"] and r["complete"])
    print(f"  {'-' * 80}")
    print(
        f"  {n_ok}/{len(rows_out)} (shape, bs) pairs: every checked structure holds; "
        f"{n_complete} of them cover every hyperpath"
    )
    out = ROOT / "results" / "validity"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "shannon_wide.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote {out / 'shannon_wide.csv'}")


if __name__ == "__main__":
    main()
