"""Run CorrBound's own LPBuilder on its 400 SM queries, as shipped and per its Lemma.

CorrBound's artifact builds CorrBound-e's LP in ``corrbound.cyclic.LPBuilder``
(OR-Tools GLOP).  Its inter-relation constraint is written
``p sum_i h(x y_i) - (k - 1) h(x) <= log2 GIP_p`` and its intra-relation one
``(p + q + 1) h(AB) - h(A) - h(B) <= log2 norm_pq`` for p in 1..3 and
p, q <= 10 with p q <= 12 respectively; the
paper's Eq. (13) and (14) give ``-(p k - 1) h(x)`` and ``- p h(A) - q h(B)``,
which agree with the code only at p = q = 1.  This runs the shipped LPBuilder
unchanged (it must reproduce the artifact's published estimates) and with the
two methods replaced by Eq. (13)/(14), on the same statistics, built here in
the artifact's formats from its DBLP files.  LpFlow (unused by CorrBound-e)
is stubbed.

Needs a clone of the artifact (env LPH_CORRBOUND_DIR, default
../external/CorrBound) and a Python with ortools, sqlparse, joblib, numpy,
scipy, pandas and duckdb (the artifact's dependencies, not this repo's).
Output: results/corrbound/sm_artifact_lp.csv and sm_artifact_lp_summary.csv
"""

from __future__ import annotations

import contextlib
import csv
import io
import multiprocessing
import os
import statistics
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parent.parent
CB = Path(os.environ.get("LPH_CORRBOUND_DIR", ROOT.parent / "external" / "CorrBound"))
PS = [*range(1, 31), np.inf]

STATS: dict[str, Any] = {}
DICT_SM: dict[str, Any] = {}
CORR: dict[str, list[list[float]]] = {}


def _norm(d: list[int], p: float) -> float:
    a = np.asarray(d or [0], dtype=float)
    return float(a.max()) if p == np.inf else float((a**p).sum() ** (1 / p))


def build_inputs() -> None:
    """The artifact's three statistics objects, from its DBLP files."""
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
    labels = sorted(set(label.values()))
    n = max(label) + 1
    by_s: dict[tuple[int, int], dict[float, float]] = {}
    by_t: dict[tuple[int, int], dict[float, float]] = {}
    rows, keys = [], {}
    for a in labels:
        for b in labels:
            es = edges.get((a, b), [])
            ds = Counter(s for s, _ in es)
            dt = Counter(t for _, t in es)
            by_s[(a, b)] = {p: _norm(list(ds.values()), p) for p in PS}
            by_t[(a, b)] = {p: _norm(list(dt.values()), p) for p in PS}
            vec = np.zeros(n, dtype=np.int64)
            for v, c in ds.items():
                vec[v] = c
            keys[(a, b)] = len(rows)
            rows.append(vec)
            CORR[f"{a}_{b}"] = [
                [p, q, float(sum(ds[s] ** p * dt[t] ** q for s, t in es))]
                for p in range(1, 11)
                for q in range(1, 11)
            ]
    STATS["dblp"] = {
        "edge": {("s", "double_constraint"): by_s, ("t", "double_constraint"): by_t}
    }
    DICT_SM["map_rows"] = keys
    DICT_SM["X_A"] = csr_matrix(np.array(rows)).astype(np.uint64)


def _lemma_inter(self: Any) -> None:
    """Eq. (13): h(x) + p sum_i h(y_i | x) <= log2 GIP_p."""
    import corrbound.cyclic.QueryParser as qp  # noqa: N813  (the artifact's name)

    x_a, map_rows = self.dict_sm_dblp["X_A"], self.dict_sm_dblp["map_rows"]
    for jv, preds in qp.get_predicate_combis(self.sql).items():
        k = len(preds)
        if k == 1:
            continue
        vecs = [
            x_a[map_rows[(int(pr[2]), int(pr[3]))], :].toarray().ravel().astype(float)
            for pr in preds
        ]
        self.add_variable_to_lp("h_" + jv)
        hx = self.lp_vars["h_" + jv]
        hxy = []
        for pr in preds:
            name = "h_" + "_".join(sorted([jv, pr[1]]))
            self.add_variable_to_lp(name)
            hxy.append(self.lp_vars[name])
        for p in (1, 2, 3):
            gip = np.ones_like(vecs[0])
            for v in vecs:
                gip = gip * v**p
            expr = -(p * k - 1) * hx
            for h in hxy:
                expr += p * h
            self.solver.Add(expr <= np.log2(gip.sum()))


def _lemma_intra(self: Any) -> None:
    """Eq. (14): h(AB) + p h(B | A) + q h(A | B) <= log2 sum d(s)^p d(t)^q."""
    import corrbound.cyclic.QueryParser as qp  # noqa: N813  (the artifact's name)

    for preds in qp.get_predicate_combis(self.sql).values():
        for pr in preds:
            a, b = pr[0], pr[1]
            h_a, h_b, h_ab = "h_" + a, "h_" + b, "h_" + "_".join(sorted([a, b]))
            for name in (h_a, h_b, h_ab):
                self.add_variable_to_lp(name)
            for p, q, norm in self.intra_stats[f"{pr[2]}_{pr[3]}"]:
                if p * q > 12:
                    continue
                self.solver.Add(
                    (p + q + 1) * self.lp_vars[h_ab]
                    - p * self.lp_vars[h_a]
                    - q * self.lp_vars[h_b]
                    <= np.log2(norm)
                )


def solve(sql: str, lemma: bool) -> float:
    import corrbound.cyclic.LPBuilder as lpb  # noqa: N813

    with contextlib.redirect_stdout(io.StringIO()):
        b = lpb.LPBuilder(
            "dblp",
            sql,
            statistics=STATS,
            inter_relation_constraints=True,
            intra_relation_constraints=True,
            dict_sm=DICT_SM,
            corr_norms=CORR,
        )
        if lemma:
            b.add_inter_relation_constraints = types.MethodType(_lemma_inter, b)
            b.add_intra_relation_constraints = types.MethodType(_lemma_intra, b)
        b.build()
        obj = b.solve()
    return float(2.0**obj)


def one(q: str) -> dict[str, str]:
    sql = (CB / "benchmarks/workloads/subgraph_matching/dblp" / f"{q}.sql").read_text()
    return {
        "query": q,
        "shipped": f"{solve(sql, False):.9g}",
        "eq13_14": f"{solve(sql, True):.9g}",
    }


def main() -> None:
    sys.path.insert(0, str(CB / "src"))
    stub = types.ModuleType("corrbound.LpFlow.flow_bound_wrapper")
    stub.py_flow_bound = None  # type: ignore[attr-defined]
    sys.modules["corrbound.LpFlow.flow_bound_wrapper"] = stub
    build_inputs()
    names = sorted(
        f[:-4]
        for f in os.listdir(CB / "benchmarks/workloads/subgraph_matching/dblp")
        if f.endswith(".sql") and "_8_" in f
    )
    with multiprocessing.get_context("fork").Pool(8) as pool:
        rows = pool.map(one, names, chunksize=1)
    with open(CB / "results/paper/accuracy_cyclic/corrbound_e_dblp.csv") as fh:
        pub = {r["query"]: float(r["estimate"]) for r in csv.DictReader(fh)}
    with open(ROOT / "results/corrbound/sm_recount.csv") as fh:
        truth = {r["query"]: float(r["count"]) for r in csv.DictReader(fh)}
    for r in rows:
        r["published"] = f"{pub[r['query']]:.9g}"
        r["truth"] = f"{truth[r['query']]:.0f}"
    out = ROOT / "results" / "corrbound"
    with open(out / "sm_artifact_lp.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def f(r: dict[str, str], c: str) -> float:
        return float(r[c])

    rep = [f(r, "shipped") / f(r, "published") for r in rows]
    gap = [f(r, "eq13_14") / f(r, "shipped") for r in rows]
    summary = {
        "n": len(rows),
        "shipped_over_published_min": f"{min(rep):.6f}",
        "shipped_over_published_max": f"{max(rep):.6f}",
        "median_eq_over_shipped": f"{statistics.median(gap):.4g}",
        "min_eq_over_shipped": f"{min(gap):.4g}",
        "max_eq_over_shipped": f"{max(gap):.4g}",
        "median_shipped_over_truth": f"{statistics.median(f(r, 'shipped') / f(r, 'truth') for r in rows):.4g}",
        "median_eq_over_truth": f"{statistics.median(f(r, 'eq13_14') / f(r, 'truth') for r in rows):.4g}",
        "n_shipped_below_truth": sum(f(r, "shipped") < f(r, "truth") for r in rows),
        "n_eq_below_truth": sum(f(r, "eq13_14") < f(r, "truth") for r in rows),
    }
    with open(out / "sm_artifact_lp_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary))
        w.writeheader()
        w.writerow(summary)
    print(summary)


if __name__ == "__main__":
    main()
