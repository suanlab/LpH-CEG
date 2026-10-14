"""Statistics vs. inference with a larger p-set on BOTH sides.

`lpbound_pset_check.py` showed that the entropic baseline gets markedly
tighter with p in {1, ..., 10, inf} than with the paper's {1, 2, inf}
(median 2.7x Hetionet, 5.0x IMDb).  A fair comparison gives the block
statistics the same norms.  For every query of both long suites this
computes, from one statistics collection with P11 = {1, ..., 10, inf},

* the entropic bound on single-relation statistics (bs=1 flow LP, which
  equals the 2^|V| program),
* the cover LP at bs=1 and bs=2, and the flow LP over the bs<=2 catalog,

for P3 = {1, 2, inf} and P11.  The P3 values must reproduce the paper's.

Outputs: results/suite/pset_fair.csv and pset_fair_summary.csv
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing
import statistics
import time
from pathlib import Path

from scripts.evaluate_imdb_suite import build_imdb_query
from scripts.evaluate_query_suite import build_hetionet_query
from scripts.flow_bound_blocks import _walk_order
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import (
    block_constraints,
    cycle_candidate_orders,
    flow_bound,
    flow_bound_over_orders,
)
from src.estimators.lph_ceg import estimate_lph_ceg
from src.statistics.lp_stats import compute_all_lp_block_extensions

ROOT = Path(__file__).resolve().parent.parent
P3 = (1.0, 2.0, math.inf)
P11 = (*(float(p) for p in range(1, 11)), math.inf)
SUITES = {
    "hetionet": (
        ROOT / "benchmarks" / "query_suite_v2_long.json",
        build_hetionet_query,
    ),
    "imdb": (ROOT / "benchmarks" / "query_suite_v3_imdb.json", build_imdb_query),
}
WORKERS = 10


def evaluate(item: tuple[str, dict]) -> dict[str, object] | None:  # type: ignore[type-arg]
    suite, q = item
    built = SUITES[suite][1](q)
    if built is None:
        return None
    query, data = built
    t0 = time.perf_counter()
    attrs = sorted(query.all_attributes)
    stats = compute_all_lp_block_extensions(
        query, enumerate_blocks(query, max_block_size=2), data, p_values=P11
    )
    orders = cycle_candidate_orders(_walk_order(query))
    row: dict[str, object] = {
        "suite": suite,
        "query_id": q["query_id"],
        "n_vars": q["n_vars"],
    }
    for tag, ps in (("p3", P3), ("p11", P11)):
        cons1 = block_constraints(
            query, data, max_block_size=1, p_values=ps, lp_stats=stats
        )
        cons2 = block_constraints(
            query, data, max_block_size=2, p_values=ps, lp_stats=stats
        )
        ent = flow_bound(attrs, cons1).bound
        c1 = estimate_lph_ceg(
            query, data, max_block_size=1, p_values=ps, lp_stats=stats
        ).estimate
        c2 = estimate_lph_ceg(
            query, data, max_block_size=2, p_values=ps, lp_stats=stats
        ).estimate
        f2 = flow_bound_over_orders(attrs, cons2, orders)[0].bound
        row.update(
            {
                f"entropic_{tag}": f"{ent:.6e}",
                f"cover_bs1_{tag}": f"{c1:.6e}",
                f"cover_bs2_{tag}": f"{c2:.6e}",
                f"flow_bs2_{tag}": f"{f2:.6e}",
            }
        )
    row["seconds"] = f"{time.perf_counter() - t0:.1f}"
    return row


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    f = lambda r, c: float(str(r[c]))  # noqa: E731
    out: list[dict[str, object]] = []
    for suite in ("hetionet", "imdb"):
        for nv in ("all", 6, 8, 10):
            rs = [
                r
                for r in rows
                if r["suite"] == suite and (nv == "all" or r["n_vars"] == nv)
            ]
            if not rs:
                continue
            rec: dict[str, object] = {"suite": suite, "n_vars": nv, "n": len(rs)}
            for tag in ("p3", "p11"):
                ent = f"entropic_{tag}"
                rec[f"median_cover_bs1_over_entropic_{tag}"] = (
                    f"{statistics.median(f(r, f'cover_bs1_{tag}') / f(r, ent) for r in rs):.6g}"
                )
                rec[f"median_cover_bs2_over_entropic_{tag}"] = (
                    f"{statistics.median(f(r, f'cover_bs2_{tag}') / f(r, ent) for r in rs):.6g}"
                )
                rec[f"median_flow_bs2_over_entropic_{tag}"] = (
                    f"{statistics.median(f(r, f'flow_bs2_{tag}') / f(r, ent) for r in rs):.6g}"
                )
                rec[f"n_cover_bs2_below_entropic_{tag}"] = sum(
                    f(r, f"cover_bs2_{tag}") < f(r, ent) for r in rs
                )
                rec[f"n_flow_bs2_below_entropic_{tag}"] = sum(
                    f(r, f"flow_bs2_{tag}") < f(r, ent) for r in rs
                )
            rec["median_entropic_p3_over_p11"] = (
                f"{statistics.median(f(r, 'entropic_p3') / f(r, 'entropic_p11') for r in rs):.6g}"
            )
            rec["median_flow_bs2_p3_over_p11"] = (
                f"{statistics.median(f(r, 'flow_bs2_p3') / f(r, 'flow_bs2_p11') for r in rs):.6g}"
            )
            out.append(rec)
    return out


def _write(name: str, rows: list[dict[str, object]]) -> None:
    with open(ROOT / "results" / "suite" / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    items = [
        (suite, q)
        for suite, (path, _) in SUITES.items()
        for q in json.loads(path.read_text())["queries"]
    ]
    rows: list[dict[str, object]] = []
    with multiprocessing.Pool(processes=WORKERS, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(evaluate, items):
            if r is None:
                continue
            rows.append(r)
            rows.sort(key=lambda x: (str(x["suite"]), str(x["query_id"])))
            _write("pset_fair.csv", rows)
            print(r["suite"], r["query_id"], r["seconds"], flush=True)
    _write("pset_fair_summary.csv", summarise(rows))
    print("done")


if __name__ == "__main__":
    main()
