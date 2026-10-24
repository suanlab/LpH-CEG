"""CorrBound-e re-implemented, and combined with block statistics.

On CorrBound's 400 DBLP subgraph-matching queries (see
``corrbound_sm_compare.py``) this adds, inside our 2^|V| polymatroid
program, the statistics CorrBound-e uses there (its artifact's
``LPBuilder`` with inter- and intra-relation constraints):

  lp      for every edge atom, the l_p norms (p = 1..30, inf) of the
          degree sequences of its two endpoints within the label-filtered
          edges (LpBound's statistics; our single-relation constraints);
  inter   for every query vertex x with k >= 2 incident edges to y_1..y_k,
          and p = 1, 2, 3, CorrBound's Eq. (13):
              h(x) + p sum_i h(y_i | x) <= log2 sum_v prod_i d_i(v)^p
          with d_i(v) the number of neighbours of v labelled like y_i
          (the k-way generalized inner product of the degree vectors);
  intra   for every edge atom (A, B) and p, q in 1..10 with p q <= 12,
          its Eq. (14):
              h(AB) + p h(B | A) + q h(A | B) <= log2 sum_(s,t) d(s)^p d(t)^q
          over the label-filtered edges (the "corr norms").

The artifact's LPBuilder writes -(k - 1) h(x) in the first and
-h(A) - h(B) in the second for every p, q; the two agree with Eq. (13)/(14)
only at p = q = 1 (``corrbound_sm_artifact_lp.py`` runs it both ways).
Here the inequalities are Eq. (13)/(14), which hold on every database.

Catalogs, all solved by the same program:

  corrbound  lp + inter + intra (Eq. 13/14) in the full polymatroid program
  block      the bs<=2 block catalog (sparse-product statistics, same norms)
  combined   block + inter + intra

Truth is the DuckDB recount (``corrbound_sm_recount.py``).
Output: results/corrbound/sm_combined.csv and sm_combined_summary.csv
"""

from __future__ import annotations

import csv
import math
import multiprocessing
import os
import statistics
import time
from collections import Counter, defaultdict

import scripts.corrbound_sm_compare as cmp
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import block_constraints, polymatroid_bound
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.statistics.pushdown_stats import compute_sparse_block_extensions

P30 = cmp.P30
WORKERS = 8
INTRA_MAX = 10
_DEG: dict[tuple[int, int], tuple[Counter[int], Counter[int]]] = {}


def degrees(la: int, lb: int) -> tuple[Counter[int], Counter[int]]:
    """Out-degree of s and in-degree of t over the edges labelled (la, lb)."""
    if (la, lb) not in _DEG:
        rows = cmp._EDGES.get((la, lb), [])
        _DEG[(la, lb)] = (Counter(s for s, _ in rows), Counter(t for _, t in rows))
    return _DEG[(la, lb)]


def corrbound_extra(
    labels: dict[int, int], edges: list[tuple[int, int]]
) -> list[tuple[dict[frozenset[str], float], float]]:
    """CorrBound-e's inter- and intra-relation constraints for one query."""
    extra: list[tuple[dict[frozenset[str], float], float]] = []
    nbrs: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    for x, ys in nbrs.items():
        k = len(ys)
        if k < 2:
            continue
        vecs = [degrees(labels[x], labels[y])[0] for y in ys]
        common = set(vecs[0]).intersection(*vecs[1:])
        for p in (1, 2, 3):
            gip = sum(math.prod(float(v[u]) ** p for v in vecs) for u in common)
            terms: dict[frozenset[str], float] = defaultdict(float)
            for y in ys:
                terms[frozenset({f"v{x}", f"v{y}"})] += p
            terms[frozenset({f"v{x}"})] -= p * k - 1
            extra.append((dict(terms), math.log2(gip) if gip > 0 else -math.inf))
    for a, b in edges:
        out_d, in_d = degrees(labels[a], labels[b])
        rows = cmp._EDGES.get((labels[a], labels[b]), [])
        for p in range(1, INTRA_MAX + 1):
            for q in range(1, INTRA_MAX + 1):
                if p * q > 12:
                    continue
                norm = sum(float(out_d[s]) ** p * float(in_d[t]) ** q for s, t in rows)
                terms = {
                    frozenset({f"v{a}", f"v{b}"}): float(p + q + 1),
                    frozenset({f"v{a}"}): -float(p),
                    frozenset({f"v{b}"}): -float(q),
                }
                extra.append((terms, math.log2(norm) if norm > 0 else -math.inf))
    return extra


def evaluate(name: str) -> dict[str, object]:
    t0 = time.perf_counter()
    sql = (
        cmp.CB
        / "benchmarks"
        / "workloads"
        / "subgraph_matching"
        / "dblp"
        / f"{name}.sql"
    ).read_text()
    labels, edges = cmp.parse(sql)
    query, data = cmp.build(sql)
    attrs = sorted(query.all_attributes)
    blocks = enumerate_blocks(query, max_block_size=2)
    stats = dict(
        compute_all_lp_block_extensions(
            query, [b for b in blocks if len(b.relations) == 1], data, p_values=P30
        )
    )
    for b in blocks:
        if len(b.relations) == 2:
            stats.update(compute_sparse_block_extensions(query, b, data, p_values=P30))
    c1 = block_constraints(query, data, max_block_size=1, p_values=P30, lp_stats=stats)
    c2 = block_constraints(query, data, max_block_size=2, p_values=P30, lp_stats=stats)
    extra = corrbound_extra(labels, edges)
    return {
        "query": name,
        "n_extra": len(extra),
        "log2_corrbound": f"{polymatroid_bound(attrs, c1, extra):.6f}",
        "log2_block": f"{polymatroid_bound(attrs, c2):.6f}",
        "log2_combined": f"{polymatroid_bound(attrs, c2, extra):.6f}",
        "seconds": f"{time.perf_counter() - t0:.1f}",
    }


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    cb = cmp._paper("corrbound_e")
    with open(cmp.OUT / "sm_recount.csv") as fh:
        truth = {r["query"]: float(r["count"]) for r in csv.DictReader(fh)}
    with open(cmp.OUT / "sm_compare.csv") as fh:
        prev = {r["query"]: float(r["log2_block_poly"]) for r in csv.DictReader(fh)}

    def e(r: dict[str, object], m: str) -> float:
        return float(2.0 ** float(str(r[f"log2_{m}"])))

    q = [str(r["query"]) for r in rows]
    rep = [e(r, "corrbound") / cb[str(r["query"])] for r in rows]
    out: list[dict[str, object]] = []
    for m in ("corrbound", "block", "combined"):
        err = [e(r, m) / truth[str(r["query"])] for r in rows]
        out.append(
            {
                "catalog": m,
                "n": len(rows),
                "median_over_truth": f"{statistics.median(err):.4g}",
                "p95_over_truth": f"{sorted(err)[int(0.95 * (len(err) - 1))]:.4g}",
                "n_below_truth": sum(x < 1 - 1e-9 for x in err),
            }
        )
    out.append(
        {
            "catalog": "checks",
            "n": len(q),
            "corrbound_ours_over_artifact_min": f"{min(rep):.6f}",
            "corrbound_ours_over_artifact_median": f"{statistics.median(rep):.6f}",
            "corrbound_ours_over_artifact_max": f"{max(rep):.6f}",
            "max_abs_log2_block_minus_compare": f"{max(abs(float(str(r['log2_block'])) - prev[str(r['query'])]) for r in rows):.2e}",
            "median_corrbound_over_block": f"{statistics.median(e(r, 'corrbound') / e(r, 'block') for r in rows):.4g}",
            "median_block_over_combined": f"{statistics.median(e(r, 'block') / e(r, 'combined') for r in rows):.4g}",
            "max_block_over_combined": f"{max(e(r, 'block') / e(r, 'combined') for r in rows):.4g}",
            "n_combined_tighter_than_block": sum(
                e(r, "block") / e(r, "combined") > 1 + 1e-6 for r in rows
            ),
            "median_corrbound_over_combined": f"{statistics.median(e(r, 'corrbound') / e(r, 'combined') for r in rows):.4g}",
        }
    )
    return out


def _write(name: str, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(cmp.OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    cmp._load_dblp()  # before the fork: workers share the index
    names = sorted(
        f[:-4]
        for f in os.listdir(
            cmp.CB / "benchmarks" / "workloads" / "subgraph_matching" / "dblp"
        )
        if f.endswith(".sql") and "_8_" in f
    )
    with multiprocessing.get_context("fork").Pool(processes=WORKERS) as pool:
        rows = pool.map(evaluate, names, chunksize=1)
    _write("sm_combined.csv", rows)
    summary = summarise(rows)
    _write("sm_combined_summary.csv", summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
