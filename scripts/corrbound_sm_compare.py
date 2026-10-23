"""Partial comparison with CorrBound on its own cyclic workload.

CorrBound (Mayer et al., SIGMOD 2026) evaluates its cyclic variant on 400
labelled subgraph-matching queries over DBLP (8 vertices; its Fig. 2), with
CorrBound-e (exact statistics) and LpBound (flow bound).  Its public
artifact (https://github.com/christophmayerm/CorrBound, MIT) ships the
queries, the data and the paper's result files, including the true
cardinalities.  This script evaluates our estimator on the same queries and
data with the same norm set as CorrBound's LpBound statistics,
P = {1, ..., 30, inf}:

  lpbound_ours  single-relation statistics, polymatroid program; must
                reproduce the artifact's LpBound estimate (sanity check)
  block_poly    + every interface of every pair block (bs <= 2), same
                program

Each edge atom e_k(s, t) becomes a binary relation over its two query
vertices holding the DBLP edges whose endpoints carry the query's labels;
the vertex atoms are key lookups with a label predicate, so folding them
into the edges preserves the count (CorrBound's LP drops them as well).
CorrBound-e's estimates and the true cardinalities are read from the
artifact's results/paper/accuracy_cyclic/.  CorrBound-e adds k-way
generalized inner products around each vertex (p = 1..3) and intra-relation
correlation norms; our bs <= 2 catalog has neither, and CorrBound has none
of the interface-conditioned pair-block norms, so this compares statistics
catalogs, each under its own authors' program.

Needs a clone of the artifact (env LPH_CORRBOUND_DIR, default
../external/CorrBound relative to the repository root).
Validity and ratios to the truth use corrbound_sm_recount.py's counts (run
it first); the summary also checks against the artifact's truth file.
Output: results/corrbound/sm_compare.csv and sm_compare_summary.csv
(``--summarise`` rewrites the summary of a finished run)
"""

from __future__ import annotations

import csv
import math
import multiprocessing
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from src.core.block import enumerate_blocks
from src.estimators.flow_bound import block_constraints, polymatroid_bound
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.utils.types import QueryHypergraph, Relation

ROOT = Path(__file__).resolve().parent.parent
CB = Path(os.environ.get("LPH_CORRBOUND_DIR", ROOT.parent / "external" / "CorrBound"))
OUT = ROOT / "results" / "corrbound"
P30 = (*(float(p) for p in range(1, 31)), math.inf)
WORKERS = 8

_LABEL = re.compile(r"v(\d+)\.l\s*=\s*(\d+)")
_ENDPOINT = re.compile(r"v(\d+)\.i\s*=\s*e(\d+)\.([st])\b")

_EDGES: dict[tuple[int, int], list[tuple[int, int]]] = {}


def _load_dblp() -> None:
    """Index the (symmetric) DBLP edge list by the labels of its endpoints."""
    label: dict[int, int] = {}
    with open(CB / "data" / "datasets" / "dblp" / "dblp_vertex.csv") as fh:
        for line in fh:
            i, lab, _ = line.split("|")
            label[int(i)] = int(lab)
    edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    with open(CB / "data" / "datasets" / "dblp" / "dblp_edge_undirected.csv") as fh:
        for line in fh:
            s, t = (int(x) for x in line.split("|"))
            edges[(label[s], label[t])].append((s, t))
    _EDGES.update(edges)


def parse(sql: str) -> tuple[dict[int, int], list[tuple[int, int]]]:
    """Vertex labels and edges (source vertex, target vertex) of an SM query."""
    labels = {int(v): int(lab) for v, lab in _LABEL.findall(sql)}
    ends: dict[int, dict[str, int]] = defaultdict(dict)
    for v, e, side in _ENDPOINT.findall(sql):
        ends[int(e)][side] = int(v)
    edges = [(ends[e]["s"], ends[e]["t"]) for e in sorted(ends)]
    return labels, edges


def build(sql: str) -> tuple[QueryHypergraph, dict]:  # type: ignore[type-arg]
    labels, edges = parse(sql)
    rels, data = [], {}
    for k, (a, b) in enumerate(edges):
        u, v = f"v{a}", f"v{b}"
        rows = _EDGES.get((labels[a], labels[b]), [])
        attrs = tuple(sorted((u, v)))
        data[f"e{k}"] = (
            attrs,
            [
                (str(s), str(t)) if attrs == (u, v) else (str(t), str(s))
                for s, t in rows
            ],
        )
        rels.append(Relation(f"e{k}", frozenset({u, v}), len(rows)))
    return QueryHypergraph(relations=tuple(rels)), data


def evaluate(name: str) -> dict[str, object]:
    t0 = time.perf_counter()
    sql = (
        CB / "benchmarks" / "workloads" / "subgraph_matching" / "dblp" / f"{name}.sql"
    ).read_text()
    query, data = build(sql)
    attrs = sorted(query.all_attributes)
    row: dict[str, object] = {"query": name, "n_rel": len(query.relations)}
    if any(not rows for _, rows in data.values()):
        row["status"] = "empty relation"
        return row
    blocks = enumerate_blocks(query, max_block_size=2)
    stats = compute_all_lp_block_extensions(query, blocks, data, p_values=P30)
    t1 = time.perf_counter()
    for name_, bs in (("lpbound_ours", 1), ("block_poly", 2)):
        cons = block_constraints(
            query, data, max_block_size=bs, p_values=P30, lp_stats=stats
        )
        row[f"log2_{name_}"] = f"{polymatroid_bound(attrs, cons):.6f}"
    row["n_pair_blocks"] = sum(len(b.relations) == 2 for b in blocks)
    row["stats_s"] = f"{t1 - t0:.2f}"
    row["solve_s"] = f"{time.perf_counter() - t1:.2f}"
    row["status"] = "ok"
    return row


def _paper(kind: str) -> dict[str, float]:
    path = CB / "results" / "paper" / "accuracy_cyclic" / f"{kind}_dblp.csv"
    with open(path) as fh:
        return {r["query"]: float(r["estimate"]) for r in csv.DictReader(fh)}


def _rows(name: str) -> int:
    with open(CB / "data" / "datasets" / "dblp" / name) as fh:
        return sum(1 for _ in fh)


def _q(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    art, cb, lpb = _paper("truecardinality"), _paper("corrbound_e"), _paper("lpbound")
    with open(OUT / "sm_recount.csv") as fh:  # corrbound_sm_recount.py
        truth = {r["query"]: float(r["count"]) for r in csv.DictReader(fh)}
    ok = [r for r in rows if r["status"] == "ok" and truth.get(str(r["query"]), 0) > 0]

    def est(r: dict[str, object], m: str) -> float:
        q = str(r["query"])
        if m == "corrbound_e":
            return cb[q]
        if m == "lpbound_artifact":
            return lpb[q]
        return 2.0 ** float(str(r[f"log2_{m}"]))

    out: list[dict[str, object]] = []
    for m in ("lpbound_artifact", "lpbound_ours", "corrbound_e", "block_poly"):
        err = [est(r, m) / truth[str(r["query"])] for r in ok]
        out.append(
            {
                "method": m,
                "n": len(ok),
                "median_over_truth": f"{statistics.median(err):.4g}",
                "p95_over_truth": f"{_q(err, 0.95):.4g}",
                "geomean_over_truth": f"{2 ** statistics.mean(math.log2(e) for e in err):.4g}",
                "n_below_truth": sum(e < 1 - 1e-9 for e in err),
                "n_below_artifact_truth": sum(
                    est(r, m) < art[str(r["query"])] * (1 - 1e-9) for r in ok
                ),
            }
        )
    ours_vs_art = [est(r, "lpbound_ours") / est(r, "lpbound_artifact") for r in ok]
    cb_over_block = [est(r, "corrbound_e") / est(r, "block_poly") for r in ok]
    lpb_over_block = [est(r, "lpbound_artifact") / est(r, "block_poly") for r in ok]
    lpb_over_cb = [est(r, "lpbound_artifact") / est(r, "corrbound_e") for r in ok]
    out.append(
        {
            "method": "pairwise",
            "n": len(ok),
            "lpbound_ours_over_artifact_min": f"{min(ours_vs_art):.6f}",
            "lpbound_ours_over_artifact_max": f"{max(ours_vs_art):.6f}",
            "median_lpbound_over_corrbound_e": f"{statistics.median(lpb_over_cb):.4g}",
            "median_lpbound_over_block_poly": f"{statistics.median(lpb_over_block):.4g}",
            "median_corrbound_e_over_block_poly": f"{statistics.median(cb_over_block):.4g}",
            "n_block_poly_tighter": sum(x > 1 + 1e-9 for x in cb_over_block),
            "n_corrbound_e_tighter": sum(x < 1 - 1e-9 for x in cb_over_block),
            "max_block_poly_over_corrbound_e": f"{1 / min(cb_over_block):.4g}",
            "median_recount_over_artifact_truth": f"{statistics.median(truth[str(r['query'])] / art[str(r['query'])] for r in ok):.4g}",
            "max_recount_over_artifact_truth": f"{max(truth[str(r['query'])] / art[str(r['query'])] for r in ok):.4g}",
            "n_dense": sum("dense" in str(r["query"]) for r in ok),
            "n_block_poly_tighter_dense": sum(
                x > 1 + 1e-9
                for r, x in zip(ok, cb_over_block, strict=True)
                if "dense" in str(r["query"])
            ),
            "n_block_poly_tighter_sparse": sum(
                x > 1 + 1e-9
                for r, x in zip(ok, cb_over_block, strict=True)
                if "sparse" in str(r["query"])
            ),
            "min_edges": min(int(str(r["n_rel"])) for r in ok),
            "max_edges": max(int(str(r["n_rel"])) for r in ok),
            "dblp_vertices": _rows("dblp_vertex.csv"),
            "dblp_edge_rows": _rows("dblp_edge_undirected.csv"),
            "median_stats_s": f"{statistics.median(float(str(r['stats_s'])) for r in ok):.3g}",
            "median_solve_s": f"{statistics.median(float(str(r['solve_s'])) for r in ok):.3g}",
            "n_truth_differs_from_artifact": sum(
                truth[str(r["query"])] != art[str(r["query"])] for r in ok
            ),
            "n_rows": len(rows),
            "n_not_ok": sum(r["status"] != "ok" for r in rows),
        }
    )
    return out


def _write(name: str, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--summarise" in sys.argv:  # re-summarise a finished run
        with open(OUT / "sm_compare.csv") as fh:
            rows = list(csv.DictReader(fh))
        _write("sm_compare_summary.csv", summarise(rows))  # type: ignore[arg-type]
        return
    _load_dblp()  # before the fork: workers share the index
    names = sorted(
        f[:-4]
        for f in os.listdir(
            CB / "benchmarks" / "workloads" / "subgraph_matching" / "dblp"
        )
        if f.endswith(".sql") and "_8_" in f
    )
    with multiprocessing.get_context("fork").Pool(processes=WORKERS) as pool:
        rows = pool.map(evaluate, names, chunksize=1)
    _write("sm_compare.csv", rows)
    summary = summarise(rows)
    _write("sm_compare_summary.csv", summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
