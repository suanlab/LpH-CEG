"""Full MOLP versus the chain max-degree baseline the paper labels CEG_M.

The paper's max-degree baseline (``estimate_pessimistic`` at bs=1) extends
one whole relation per step through a non-empty interface.  MOLP
(Joglekar and Re) is the shortest path over *all* attribute subsets, with,
for every binary relation R(u, v),

  S -> S + {u, v}   cost log2 |R|          (from any S)
  S -> S + {u}      cost log2 |pi_u R|     (projections)
  S -> S + {v}      cost log2 max_u deg_R(v | u)   (u in S)

and the symmetric edges.  It can only be tighter than the chain (the chain's
steps are among its edges).  This recomputes the synthetic cycle table
(cycle-3..8, 200 vertices, 800 edges, power-law alpha = 1, the same ten
seeds as ``revision_experiments.py``) with full MOLP next to the chain and
the bs=1 cover LP.

Output: results/revision/molp_full.csv and molp_full_summary.csv
"""

from __future__ import annotations

import csv
import heapq
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.revision_experiments import P_VALUES, make_cycle_data
from src.estimators.lpbound_baseline import estimate_cover_bs1
from src.estimators.pessimistic import estimate_pessimistic

if TYPE_CHECKING:
    from src.statistics.block_stats import RelationDataMap
    from src.utils.types import QueryHypergraph

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "revision"
SEEDS = [1000 + s * 100 for s in range(10)]


def full_molp(query: QueryHypergraph, data: RelationDataMap) -> float:
    """MOLP bound of a query over binary relations (log2 shortest path)."""
    attrs = sorted(query.all_attributes)
    bit = {a: 1 << i for i, a in enumerate(attrs)}
    edges: list[tuple[int, int, float]] = []  # (attributes needed, added, cost)
    for rel in query.relations:
        schema, rows = data[rel.name]
        rows = list(dict.fromkeys(rows))
        if not rows:
            return -math.inf
        u, v = schema
        iu, iv = schema.index(u), schema.index(v)
        deg_v: dict[str, set[str]] = defaultdict(set)
        deg_u: dict[str, set[str]] = defaultdict(set)
        for r in rows:
            deg_v[r[iu]].add(r[iv])
            deg_u[r[iv]].add(r[iu])
        bu, bv = bit[u], bit[v]
        edges.append((0, bu | bv, math.log2(len(rows))))
        edges.append((0, bu, math.log2(len(deg_v))))
        edges.append((0, bv, math.log2(len(deg_u))))
        edges.append((bu, bv, math.log2(max(len(s) for s in deg_v.values()))))
        edges.append((bv, bu, math.log2(max(len(s) for s in deg_u.values()))))
    full = (1 << len(attrs)) - 1
    dist = {0: 0.0}
    heap = [(0.0, 0)]
    while heap:
        d, s = heapq.heappop(heap)
        if s == full:
            return d
        if d > dist.get(s, math.inf):
            continue
        for need, add, cost in edges:
            if s & need != need or s | add == s:
                continue
            t, nd = s | add, d + cost
            if nd < dist.get(t, math.inf):
                dist[t] = nd
                heapq.heappush(heap, (nd, t))
    return math.inf


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for k in range(3, 9):
        for seed in SEEDS:
            q, data = make_cycle_data(k, 200, 800, alpha=1.0, seed=seed)
            chain = estimate_pessimistic(q, data, max_block_size=1).estimate
            cover = estimate_cover_bs1(q, data, p_values=P_VALUES).estimate
            molp = 2.0 ** full_molp(q, data)
            rows.append(
                {
                    "query": f"cycle_{k}",
                    "seed": seed,
                    "chain": f"{chain:.6e}",
                    "full_molp": f"{molp:.6e}",
                    "cover_bs1": f"{cover:.6e}",
                }
            )
            print(rows[-1], flush=True)
    with open(OUT / "molp_full.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for k in range(3, 9):
        rs = [r for r in rows if r["query"] == f"cycle_{k}"]

        def f(r: dict[str, object], c: str) -> float:
            return float(str(r[c]))

        summary.append(
            {
                "query": f"cycle_{k}",
                "n": len(rs),
                "mean_chain_over_cover": f"{statistics.mean(f(r, 'chain') / f(r, 'cover_bs1') for r in rs):.4g}",
                "mean_molp_over_cover": f"{statistics.mean(f(r, 'full_molp') / f(r, 'cover_bs1') for r in rs):.4g}",
                "mean_chain_over_molp": f"{statistics.mean(f(r, 'chain') / f(r, 'full_molp') for r in rs):.4g}",
                "max_chain_over_molp": f"{max(f(r, 'chain') / f(r, 'full_molp') for r in rs):.4g}",
            }
        )
    with open(OUT / "molp_full_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(s)


def suites() -> None:
    """Full MOLP on the Hetionet and IMDb long suites, beside their stored
    chain (``maxdeg``) and bs=1 cover bounds."""
    import json

    from scripts.pset_fair import SUITES

    stored = {}
    for name, path in (
        ("hetionet", ROOT / "results" / "suite" / "long_suite_eval.csv"),
        ("imdb", ROOT / "results" / "imdb" / "imdb_suite_eval.csv"),
    ):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                stored[(name, r["query_id"])] = r
    rows: list[dict[str, object]] = []
    for suite in ("hetionet", "imdb"):
        for q in json.loads(SUITES[suite][0].read_text())["queries"]:
            built = SUITES[suite][1](q)
            if built is None:
                continue
            query, data = built
            r = stored[(suite, q["query_id"])]
            rows.append(
                {
                    "suite": suite,
                    "query_id": q["query_id"],
                    "n_vars": q["n_vars"],
                    "chain": r["maxdeg"],
                    "full_molp": f"{2.0 ** full_molp(query, data):.6e}",
                    "cover_bs1": r["cover_bs1"],
                }
            )
    with open(OUT / "molp_full_suites.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = []
    for suite in ("hetionet", "imdb"):
        for nv in ("6", "8", "10"):
            rs = [r for r in rows if r["suite"] == suite and str(r["n_vars"]) == nv]

            def g(r: dict[str, object], c: str) -> float:
                return float(str(r[c]))

            out.append(
                {
                    "suite": suite,
                    "n_vars": nv,
                    "n": len(rs),
                    "median_chain_over_cover": f"{statistics.median(g(r, 'chain') / g(r, 'cover_bs1') for r in rs):.4g}",
                    "median_molp_over_cover": f"{statistics.median(g(r, 'full_molp') / g(r, 'cover_bs1') for r in rs):.4g}",
                    "median_chain_over_molp": f"{statistics.median(g(r, 'chain') / g(r, 'full_molp') for r in rs):.4g}",
                    "n_molp_below_cover": sum(
                        g(r, "full_molp") < g(r, "cover_bs1") * (1 - 1e-9) for r in rs
                    ),
                }
            )
    with open(OUT / "molp_full_suites_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for s in out:
        print(s)


if __name__ == "__main__":
    import sys

    suites() if "--suites" in sys.argv else main()
