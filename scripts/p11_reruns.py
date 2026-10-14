"""Rerun the remaining P3-only comparisons with ten norms on both sides.

After `pset_fair.py` (long suites), three comparisons still used only
P3 = {1, 2, inf}: the 109-query short Hetionet suite (3K-row prefixes),
the 12 synthetic family/seed pairs of the statistics-vs-inference 2x2, and
the synthetic cycle sweep behind the cover-vs-entropic gap.  For each this
computes, from one statistics collection with P11 = {1, ..., 10, inf}, the
cover LP (bs=1, bs=2), the entropic bound (bs=1 flow LP), and the flow LP
over the bs<=2 catalog (plus the 2^|V| polymatroid bound on the families),
for both P3 and P11.

Outputs: results/suite/p11_short.csv, results/flow/p11_2x2.csv,
results/scalability/p11_sweep.csv
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing
from pathlib import Path

from scripts import real_frontier as rf
from scripts import statistics_hierarchy as sh
from scripts.flow_bound_blocks import FAMILIES, SEEDS, _orders, _walk_order
from scripts.scalability_vs_entropic import COVER_K_VALUES, _build_cycle
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import (
    block_constraints,
    cycle_candidate_orders,
    flow_bound,
    flow_bound_over_orders,
    polymatroid_bound,
)
from src.estimators.lph_ceg import estimate_lph_ceg
from src.statistics.lp_stats import compute_all_lp_block_extensions

ROOT = Path(__file__).resolve().parent.parent
P3 = (1.0, 2.0, math.inf)
P11 = (*(float(p) for p in range(1, 11)), math.inf)


def bounds(query, data, orders, *, bs2: bool, poly: bool) -> dict[str, str]:  # type: ignore[no-untyped-def]
    attrs = sorted(query.all_attributes)
    mbs = 2 if bs2 else 1
    stats = compute_all_lp_block_extensions(
        query, enumerate_blocks(query, max_block_size=mbs), data, p_values=P11
    )
    out: dict[str, str] = {}
    for tag, ps in (("p3", P3), ("p11", P11)):
        cons1 = block_constraints(
            query, data, max_block_size=1, p_values=ps, lp_stats=stats
        )
        out[f"entropic_{tag}"] = f"{flow_bound(attrs, cons1).bound:.6e}"
        c1 = estimate_lph_ceg(
            query, data, max_block_size=1, p_values=ps, lp_stats=stats
        )
        out[f"cover_bs1_{tag}"] = f"{c1.estimate:.6e}"
        if bs2:
            cons2 = block_constraints(
                query, data, max_block_size=2, p_values=ps, lp_stats=stats
            )
            c2 = estimate_lph_ceg(
                query, data, max_block_size=2, p_values=ps, lp_stats=stats
            )
            out[f"cover_bs2_{tag}"] = f"{c2.estimate:.6e}"
            out[f"flow_bs2_{tag}"] = (
                f"{flow_bound_over_orders(attrs, cons2, orders)[0].bound:.6e}"
            )
            if poly:
                out[f"poly_bs2_{tag}"] = f"{2.0 ** polymatroid_bound(attrs, cons2):.6e}"
    return out


def short_case(q: dict) -> dict[str, object] | None:  # type: ignore[type-arg]
    built = rf.build(q)
    if built is None:
        return None
    query, data = built
    b = bounds(
        query, data, cycle_candidate_orders(_walk_order(query)), bs2=True, poly=False
    )
    return {"query_id": q["query_id"], "shape": q["shape"], **b}


def family_case(item: tuple[str, int]) -> dict[str, object]:
    fam, seed = item
    q, d = sh.build(sh.FAMILIES[fam], seed=seed)
    return {
        "family": fam,
        "seed": seed,
        **bounds(q, d, _orders(q), bs2=True, poly=True),
    }


def sweep_case(k: int) -> dict[str, object]:
    q, d = _build_cycle(k, 100, 400)
    return {"k": k, **bounds(q, d, [], bs2=False, poly=False)}


def _write(rel: str, rows: list[dict[str, object]]) -> None:
    rows = [r for r in rows if r is not None]
    with open(ROOT / rel, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {rel} ({len(rows)} rows)", flush=True)


def main() -> None:
    reg = json.loads(rf.SUITE.read_text())
    short = [
        q
        for q in reg["queries"]
        if q["dataset"] == "hetionet" and q["n_vars"] <= rf.MAX_VARS
    ]
    with multiprocessing.Pool(processes=10, maxtasksperchild=1) as pool:
        _write(
            "results/flow/p11_2x2.csv",
            pool.map(family_case, [(f, s) for f in FAMILIES for s in SEEDS]),
        )
        _write(
            "results/scalability/p11_sweep.csv", pool.map(sweep_case, COVER_K_VALUES)
        )
        _write("results/suite/p11_short.csv", pool.map(short_case, short))


if __name__ == "__main__":
    main()
