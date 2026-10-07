"""AGM over the query enriched with materialised 2-way join sizes.

The natural objection to the bs=2 result is that materialising pairwise
joins and then bounding is a known trick: add each join R |x| S as a
hyperedge of size |R |x| S| and run AGM.  That baseline uses the same
materialised joins as bs=2 but keeps only their *size*, not their
interface-conditioned degree sequence.  If it matches bs=2, the block
statistic is just the join size; if bs=2 is tighter, the conditioning is
what carries the gain.

Reads the pre-registered long suite and `results/suite/long_suite_eval.csv`
(for cover_bs2 and entropic on the same queries); writes
`results/suite/agm_pairjoin.csv` and `agm_pairjoin_summary.csv`.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from scipy.optimize import linprog

from scripts.evaluate_query_suite import build_hetionet_query
from src.estimators.agm_bound import estimate_agm

if TYPE_CHECKING:
    from src.utils.types import QueryHypergraph, RelationDataMap

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / "benchmarks" / "query_suite_v2_long.json"
EVAL = ROOT / "results" / "suite" / "long_suite_eval.csv"


def pair_join_size(data: RelationDataMap, r: str, s: str) -> int:
    (ca, ra), (cb, rb) = data[r], data[s]
    common = [c for c in ca if c in cb]
    ia = [ca.index(c) for c in common]
    ib = [cb.index(c) for c in common]
    ka = Counter(tuple(t[i] for i in ia) for t in ra)
    kb = Counter(tuple(t[i] for i in ib) for t in rb)
    return sum(n * kb[k] for k, n in ka.items())


def agm_with_pair_joins(query: QueryHypergraph, data: RelationDataMap) -> float:
    rels = list(query.relations)
    attrs = sorted({a for r in rels for a in r.attributes})
    edges: list[tuple[frozenset[str], float]] = [
        (frozenset(r.attributes), math.log2(max(len(data[r.name][1]), 1))) for r in rels
    ]
    for r, s in itertools.combinations(rels, 2):
        if r.attributes & s.attributes:
            size = pair_join_size(data, r.name, s.name)
            edges.append(
                (frozenset(r.attributes | s.attributes), math.log2(max(size, 1)))
            )
    c = [w for _, w in edges]
    a_ub = [[-1.0 if a in e else 0.0 for e, _ in edges] for a in attrs]
    res = linprog(
        c,
        A_ub=a_ub,
        b_ub=[-1.0] * len(attrs),
        bounds=[(0, None)] * len(edges),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(res.message)
    return float(2.0**res.fun)


def main() -> None:
    with open(SUITE) as fh:
        suite = json.load(fh)
    with open(EVAL) as fh:
        prior = {r["query_id"]: r for r in csv.DictReader(fh)}
    rows: list[dict[str, object]] = []
    print(
        f"  {'query':<9} {'|V|':>3} {'AGM':>10} {'AGM+pairs':>10} {'bs2':>10} "
        f"{'entropic':>10} {'bs2/AGM+p':>9} {'AGM+p/ent':>9}"
    )
    for q in suite["queries"]:
        built = build_hetionet_query(q)
        if built is None:
            continue
        query, data = built
        agm = estimate_agm(query, data)
        agm_p = agm_with_pair_joins(query, data)
        p = prior[q["query_id"]]
        bs2 = float(p["cover_bs2"])
        ent = float(p["entropic"])
        rows.append(
            {
                "query_id": q["query_id"],
                "n_vars": len(q["relations"]),
                "agm": f"{agm:.6e}",
                "agm_pairjoins": f"{agm_p:.6e}",
                "cover_bs2": f"{bs2:.6e}",
                "entropic": f"{ent:.6e}",
                "bs2_over_agm_pairjoins": f"{bs2 / agm_p:.6f}",
                "agm_pairjoins_over_entropic": f"{agm_p / ent:.6f}",
            }
        )
        print(
            f"  {q['query_id']:<9} {len(q['relations']):>3} {agm:>10.2e} "
            f"{agm_p:>10.2e} {bs2:>10.2e} {ent:>10.2e} {bs2 / agm_p:>9.3f} "
            f"{agm_p / ent:>9.3f}",
            flush=True,
        )

    out = ROOT / "results" / "suite"
    with open(out / "agm_pairjoin.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary: list[dict[str, object]] = []
    for k in sorted({int(str(r["n_vars"])) for r in rows}):
        rs = [r for r in rows if int(str(r["n_vars"])) == k]
        x = [float(str(r["bs2_over_agm_pairjoins"])) for r in rs]
        y = [float(str(r["agm_pairjoins_over_entropic"])) for r in rs]
        n_below = sum(1 for v in x if v < 1.0)
        summary.append(
            {
                "n_vars": k,
                "n_queries": len(rs),
                "median_bs2_over_agm_pairjoins": f"{statistics.median(x):.6f}",
                "max_bs2_over_agm_pairjoins": f"{max(x):.6f}",
                "n_bs2_below_agm_pairjoins": n_below,
                "median_agm_pairjoins_over_entropic": f"{statistics.median(y):.6f}",
                "n_agm_pairjoins_below_entropic": sum(1 for v in y if v < 1.0),
            }
        )
        print(
            f"  |V|={k}: bs2/AGM+pairs median {statistics.median(x):.3f} "
            f"(max {max(x):.3f}, below 1 on {n_below}/{len(rs)}); "
            f"AGM+pairs/entropic median {statistics.median(y):.3f}"
        )
    with open(out / "agm_pairjoin_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nWrote {out / 'agm_pairjoin.csv'} and agm_pairjoin_summary.csv")


if __name__ == "__main__":
    main()
