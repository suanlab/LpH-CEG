"""Certify the G-CARE flow-LP bounds (eleven norms) by the polymatroid bound.

Companion of ``poly_certify.py`` for the G-CARE suite: for every query
``gcare_suite.py`` evaluated, rebuild its relations, recollect the P11
statistics, and compare the order-relaxed flow LP over the ``bs<=2``
catalog with the ``2**|V|`` polymatroid bound over the unrelaxed catalog,
which is valid for any p.  ``flow >= poly`` certifies a query's flow
bound.  The flow values must reproduce ``gcare_eval.csv``.

Outputs: results/gcare/gcare_poly_certify.csv and
gcare_poly_certify_summary.csv
"""

from __future__ import annotations

import csv
import multiprocessing
import statistics
import sys
import time

import duckdb

from scripts.gcare_suite import (
    DUCKDB_THREADS,
    GCARE,
    OUT,
    P11,
    _limit_memory,
    parse,
    rel_sql,
    walk_orders,
)
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import (
    block_constraints,
    flow_bound_over_orders,
    polymatroid_bound,
)
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.utils.types import QueryHypergraph, Relation

WORKERS = 8


def evaluate(q: dict[str, str]) -> dict[str, object]:
    t0 = time.perf_counter()
    ds = q["dataset"]
    verts, edges = parse(GCARE / "queries" / q["path"])
    con = duckdb.connect(str(GCARE / "gcare.duckdb"), read_only=True)
    con.execute(f"SET threads TO {DUCKDB_THREADS}")
    rels, data, vedges = [], {}, []
    for i, (a, b, lab) in enumerate(edges):
        name, u, v = f"R{i}", f"v{a}", f"v{b}"
        rows = con.execute(rel_sql(ds, verts, a, b, lab)).fetchall()
        attrs = tuple(sorted((u, v)))
        data[name] = (
            attrs,
            [(str(x), str(y)) for x, y in rows]
            if attrs == (u, v)
            else [(str(y), str(x)) for x, y in rows],
        )
        rels.append(Relation(name, frozenset({u, v}), len(rows)))
        vedges.append((u, v))
    con.close()
    query = QueryHypergraph(relations=tuple(rels), name=q["path"])
    attrs = sorted(query.all_attributes)
    stats = compute_all_lp_block_extensions(
        query, enumerate_blocks(query, max_block_size=2), data, p_values=P11
    )
    cons2 = block_constraints(
        query, data, max_block_size=2, p_values=P11, lp_stats=stats
    )
    flow = flow_bound_over_orders(attrs, cons2, walk_orders(attrs, vedges))[0].bound
    t1 = time.perf_counter()
    poly = 2.0 ** polymatroid_bound(attrs, cons2)
    return {
        "dataset": ds,
        "path": q["path"],
        "n_vars": q["n_vars"],
        "flow_bs2_p11": f"{flow:.6e}",
        "poly_bs2_p11": f"{poly:.6e}",
        "flow_over_poly_p11": f"{flow / poly:.6f}",
        "poly_s": f"{time.perf_counter() - t1:.2f}",
        "seconds": f"{time.perf_counter() - t0:.1f}",
    }


def _safe(q: dict[str, str]) -> dict[str, object]:
    _limit_memory()
    try:
        return evaluate(q)
    except MemoryError:
        return {"dataset": q["dataset"], "path": q["path"], "n_vars": q["n_vars"]}


def _write(name: str, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ev = {
        r["path"]: float(r["flow_bs2_p11"])
        for r in csv.DictReader((OUT / "gcare_eval.csv").read_text().splitlines())
        if r.get("flow_bs2_p11")
    }
    out: list[dict[str, object]] = []
    for ds in ("aids", "yago"):
        rs = [r for r in rows if r["dataset"] == ds and r.get("flow_over_poly_p11")]
        if not rs:
            continue
        ratio = [float(str(r["flow_over_poly_p11"])) for r in rs]
        repro = [float(str(r["flow_bs2_p11"])) / ev[str(r["path"])] for r in rs]
        out.append(
            {
                "dataset": ds,
                "n": len(rs),
                "n_failed": sum(
                    1
                    for r in rows
                    if r["dataset"] == ds and not r.get("flow_over_poly_p11")
                ),
                "n_flow_below_poly": sum(x < 1 - 1e-6 for x in ratio),
                "n_flow_eq_poly": sum(abs(x - 1) <= 1e-6 for x in ratio),
                "median_flow_over_poly": f"{statistics.median(ratio):.6g}",
                "max_flow_over_poly": f"{max(ratio):.6g}",
                "max_flow_vs_gcare_eval": f"{max(abs(x - 1) for x in repro):.3g}",
                "max_poly_s": max(float(str(r["poly_s"])) for r in rs),
            }
        )
    return out


def main() -> None:
    path = OUT / "gcare_poly_certify.csv"
    rows: list[dict[str, object]] = []
    if "--resume" in sys.argv and path.exists():
        rows = list(csv.DictReader(path.read_text().splitlines()))
    if "--summary-only" not in sys.argv:
        done = {r["path"] for r in rows}
        todo = [
            r
            for r in csv.DictReader((OUT / "gcare_eval.csv").read_text().splitlines())
            if r.get("cover_bs1_p11") and r["path"] not in done
        ]
        with multiprocessing.Pool(processes=WORKERS, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(_safe, todo):
                rows.append(r)
                rows.sort(key=lambda x: str(x["path"]))
                _write("gcare_poly_certify.csv", rows)
                print(
                    r["path"], r.get("flow_over_poly_p11"), r.get("seconds"), flush=True
                )
    _write("gcare_poly_certify_summary.csv", summarise(rows))
    print("done")


if __name__ == "__main__":
    main()
