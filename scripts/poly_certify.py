"""Certify every real-data flow-LP bound with eleven norms by the polymatroid bound.

The flow LP over order-relaxed ``bs=2`` constraints is proved valid for
l_inf constraints (Im et al., Thm 1.5(a)) but not for l_p ones.  The
``2**|V|`` polymatroid bound over the *unrelaxed* catalog is valid for any
p (every l_p constraint holds on the entropy of the join's uniform
distribution, LpBound), so ``flow >= poly`` on a query certifies that
query's flow bound.  ``pset_fair.py`` gives the eleven-norm (P11) flow bounds
of both long suites but not the polymatroid bound; this recomputes the
statistics and adds it.  The flow values must reproduce ``pset_fair.csv``.

With --short, the 109-query short Hetionet suite (``p11_short.csv``).

Outputs: results/suite/poly_certify{,_short}.csv and poly_certify_summary.csv
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import statistics
import sys
import time
from pathlib import Path

from scripts import real_frontier as rf
from scripts.flow_bound_blocks import _walk_order
from scripts.pset_fair import P11, SUITES
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import (
    block_constraints,
    cycle_candidate_orders,
    flow_bound_over_orders,
    polymatroid_bound,
)
from src.statistics.lp_stats import compute_all_lp_block_extensions

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "suite"
WORKERS = 10


def evaluate(item: tuple[str, dict]) -> dict[str, object] | None:  # type: ignore[type-arg]
    suite, q = item
    built = rf.build(q) if suite == "short" else SUITES[suite][1](q)
    if built is None:
        return None
    query, data = built
    t0 = time.perf_counter()
    attrs = sorted(query.all_attributes)
    stats = compute_all_lp_block_extensions(
        query, enumerate_blocks(query, max_block_size=2), data, p_values=P11
    )
    cons2 = block_constraints(
        query, data, max_block_size=2, p_values=P11, lp_stats=stats
    )
    orders = cycle_candidate_orders(_walk_order(query))
    flow = flow_bound_over_orders(attrs, cons2, orders)[0].bound
    t1 = time.perf_counter()
    poly = 2.0 ** polymatroid_bound(attrs, cons2)
    return {
        "suite": suite,
        "query_id": q["query_id"],
        "n_vars": q["n_vars"],
        "flow_bs2_p11": f"{flow:.6e}",
        "poly_bs2_p11": f"{poly:.6e}",
        "flow_over_poly_p11": f"{flow / poly:.6f}",
        "poly_s": f"{time.perf_counter() - t1:.2f}",
        "seconds": f"{time.perf_counter() - t0:.1f}",
    }


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fair = {
        (r["suite"], r["query_id"]): float(r["flow_bs2_p11"])
        for r in csv.DictReader((OUT / "pset_fair.csv").read_text().splitlines())
    }
    fair.update(
        {
            ("short", r["query_id"]): float(r["flow_bs2_p11"])
            for r in csv.DictReader((OUT / "p11_short.csv").read_text().splitlines())
        }
    )
    out: list[dict[str, object]] = []
    for suite in ("short", "hetionet", "imdb"):
        rs = [r for r in rows if r["suite"] == suite]
        if not rs:
            continue
        ratio = [float(str(r["flow_over_poly_p11"])) for r in rs]
        repro = [
            float(str(r["flow_bs2_p11"])) / fair[(suite, str(r["query_id"]))]
            for r in rs
        ]
        out.append(
            {
                "suite": suite,
                "n": len(rs),
                "n_flow_below_poly": sum(x < 1 - 1e-6 for x in ratio),
                "n_flow_eq_poly": sum(abs(x - 1) <= 1e-6 for x in ratio),
                "median_flow_over_poly": f"{statistics.median(ratio):.6g}",
                "max_flow_over_poly": f"{max(ratio):.6g}",
                "max_flow_vs_pset_fair": f"{max(abs(x - 1) for x in repro):.3g}",
                "max_poly_s": max(float(str(r["poly_s"])) for r in rs),
            }
        )
    return out


def _write(name: str, rows: list[dict[str, object]]) -> None:
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    short = "--short" in sys.argv
    name = "poly_certify_short.csv" if short else "poly_certify.csv"
    path = OUT / name
    rows: list[dict[str, object]] = []
    if "--resume" in sys.argv and path.exists():
        rows = list(csv.DictReader(path.read_text().splitlines()))
    if "--summary-only" not in sys.argv:
        done = {(r["suite"], r["query_id"]) for r in rows}
        items = (
            [
                ("short", q)
                for q in json.loads(rf.SUITE.read_text())["queries"]
                if q["dataset"] == "hetionet" and q["n_vars"] <= rf.MAX_VARS
            ]
            if short
            else [
                (suite, q)
                for suite, (qpath, _) in SUITES.items()
                for q in json.loads(qpath.read_text())["queries"]
            ]
        )
        items = [it for it in items if (it[0], it[1]["query_id"]) not in done]
        with multiprocessing.Pool(processes=WORKERS, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(evaluate, items):
                if r is None:
                    continue
                rows.append(r)
                rows.sort(key=lambda x: (str(x["suite"]), str(x["query_id"])))
                _write(name, rows)
                print(
                    r["suite"],
                    r["query_id"],
                    r["flow_over_poly_p11"],
                    r["seconds"],
                    flush=True,
                )
    if short or (OUT / "poly_certify_short.csv").exists():
        other = OUT / ("poly_certify.csv" if short else "poly_certify_short.csv")
        if other.exists():
            rows += list(csv.DictReader(other.read_text().splitlines()))
    _write("poly_certify_summary.csv", summarise(rows))
    print("done")


if __name__ == "__main__":
    main()
