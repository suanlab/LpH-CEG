"""B3: flow-bound inference over block statistics.

Three measurements, all feeding ``results/flow/``:

1. ``flow_2x2.csv`` -- the statistics-vs-inference grid of
   ``statistics_vs_inference.py`` with a third inference column.  At ``bs=1``
   the flow LP must equal the ``2**|V|`` entropic LP (Zhang et al.,
   Thm. 4.5).  At ``bs=2`` it is compared with the polymatroid bound over
   the *identical* block constraints (the exact but exponential reference)
   and with the cover LP.
2. ``flow_long_suite.csv`` -- the pre-registered long-cycle suite
   (``|V| in {6, 8, 10}``, Hetionet, 120 queries), flow-on-blocks against
   cover-on-blocks, the polymatroid reference, LpBound's entropic bound on
   single-relation statistics, and the true count.  Statistics come from
   the amortised catalogue so the run is dominated by inference.
3. ``flow_scale.csv`` -- solve time of the flow LP on the cycle sweep of
   ``scalability_vs_entropic.py`` (``k = 3..16``) at ``bs=1`` and ``bs=2``,
   so the cost figure can show the polynomial inference LpBound actually
   ships next to the exponential formulation we measured before.

Non-simple ``|I| = 2`` block constraints are relaxed along an attribute
order; on cycles the candidates are the ``2|V|`` walks around the cycle
plus the ``2|V|`` alternating (bipartite) orders, and on the small
synthetic families all permutations; the best order is kept.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING

from scripts import statistics_hierarchy as sh
from scripts.evaluate_query_suite import P_VALUES, build_hetionet_query
from scripts.scalability_vs_entropic import COVER_K_VALUES, _build_cycle
from scripts.stats_catalogue import EVAL, SUITE, StatsCatalogue, relation_sources
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import (
    block_constraints,
    cycle_candidate_orders,
    flow_bound,
    flow_bound_over_orders,
    polymatroid_bound,
)
from src.estimators.lpbound_entropic import estimate_lpbound_entropic
from src.estimators.lph_ceg import estimate_lph_ceg
from src.statistics.lp_stats import compute_all_lp_block_extensions

if TYPE_CHECKING:
    from src.utils.types import QueryHypergraph

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "flow"
FAMILIES = ("triangle", "cycle4", "diamond", "k4")
SEEDS = (101, 202, 303)
#: Largest |V| for which the 2^|V| polymatroid reference is solved.
POLY_MAX_VARS = 10
N_TIMING_RUNS = 5


def _orders(query: QueryHypergraph) -> list[tuple[str, ...]]:
    attrs = sorted(query.all_attributes)
    if len(attrs) <= 5:
        return [tuple(p) for p in itertools.permutations(attrs)]
    return cycle_candidate_orders(attrs)


def _walk_order(query: QueryHypergraph) -> tuple[str, ...]:
    """Attribute order following the relations of a cycle query (v0, v1, ...)."""
    attrs = sorted(query.all_attributes, key=lambda a: int(a[1:]))
    return tuple(attrs)


def part_2x2() -> None:
    rows: list[dict[str, object]] = []
    print("1. Statistics x inference grid with the flow LP")
    print(
        f"  {'family':<9} {'seed':>4} {'ent1':>10} {'flow1':>10} {'cover2':>10} "
        f"{'flowS2':>10} {'flow2':>10} {'poly2':>10} {'exact':>9}"
    )
    for fam in FAMILIES:
        for seed in SEEDS:
            q, d = sh.build(sh.FAMILIES[fam], seed=seed)
            attrs = sorted(q.all_attributes)
            cons1 = block_constraints(q, d, max_block_size=1, p_values=sh.P_VALUES)
            cons2 = block_constraints(q, d, max_block_size=2, p_values=sh.P_VALUES)
            ent1 = estimate_lpbound_entropic(q, d, p_values=sh.P_VALUES)
            f1 = flow_bound(attrs, cons1)
            c1 = estimate_lph_ceg(q, d, max_block_size=1, p_values=sh.P_VALUES)
            c2 = estimate_lph_ceg(q, d, max_block_size=2, p_values=sh.P_VALUES)
            fs2 = flow_bound(attrs, cons2)
            orders = _orders(q)
            f2, best_order = flow_bound_over_orders(attrs, cons2, orders)
            t0 = time.perf_counter()
            poly2 = polymatroid_bound(attrs, cons2)
            poly_s = time.perf_counter() - t0
            poly_simple = polymatroid_bound(attrs, [c for c in cons2 if c.is_simple])
            exact = sh.true_count(q, d)
            rows.append(
                {
                    "family": fam,
                    "seed": seed,
                    "n_vars": len(attrs),
                    "n_constraints_bs2": len(cons2),
                    "n_orders": len(orders),
                    "cover_bs1": f"{c1.estimate:.6e}",
                    "entropic_bs1": f"{ent1:.6e}",
                    "flow_bs1": f"{f1.bound:.6e}",
                    "cover_bs2": f"{c2.estimate:.6e}",
                    "flow_bs2_simple": f"{fs2.bound:.6e}",
                    "poly_bs2_simple": f"{2.0**poly_simple:.6e}",
                    "flow_bs2": f"{f2.bound:.6e}",
                    "poly_bs2": f"{2.0**poly2:.6e}",
                    "true_count": exact,
                    "best_order": "".join(best_order) if best_order else "",
                    "flow_bs1_eq_entropic": abs(f1.log_bound - math.log2(ent1)) < 1e-6,
                    "flow_bs2_over_poly_bs2": f"{f2.bound / 2.0**poly2:.4f}",
                    "cover_bs2_over_flow_bs2": f"{c2.estimate / f2.bound:.4f}",
                    "entropic_bs1_over_flow_bs2": f"{ent1 / f2.bound:.4f}",
                    "flow_bs2_over_true": f"{f2.bound / exact:.4f}",
                    "flow_bs1_ms": f"{f1.solve_seconds * 1e3:.2f}",
                    "flow_bs2_ms_per_order": f"{f2.solve_seconds * 1e3 / (len(orders) + 1):.2f}",
                    "flow_bs2_ms_total": f"{f2.solve_seconds * 1e3:.1f}",
                    "poly_bs2_ms": f"{poly_s * 1e3:.1f}",
                    "flow_bs2_vars": f2.n_variables,
                }
            )
            print(
                f"  {fam:<9} {seed:>4} {ent1:>10.1f} {f1.bound:>10.1f} "
                f"{c2.estimate:>10.1f} {fs2.bound:>10.1f} {f2.bound:>10.1f} "
                f"{2.0**poly2:>10.1f} {exact:>9d}",
                flush=True,
            )
    _write("flow_2x2.csv", rows)


def part_long_suite() -> None:
    with open(SUITE) as fh:
        suite = json.load(fh)
    with open(EVAL) as fh:
        prior = {r["query_id"]: r for r in csv.DictReader(fh)}
    cat = StatsCatalogue(P_VALUES)
    rows: list[dict[str, object]] = []
    print("2. Long-cycle suite: flow-on-blocks vs cover-on-blocks vs polymatroid")
    print(
        f"  {'query':<9} {'|V|':>3} {'cover2':>10} {'flow2':>10} {'poly2':>10} "
        f"{'ent1':>10} {'true':>9} {'c2/f2':>6} {'f2/p2':>6} {'flow_s':>7} "
        f"{'poly_s':>7}"
    )
    for q in suite["queries"]:
        built = build_hetionet_query(q)
        if built is None:
            continue
        query, data = built
        attrs = sorted(query.all_attributes)
        sources = relation_sources(q)
        blocks = enumerate_blocks(query, max_block_size=2)
        stats, _, _ = cat.stats_for(query, data, blocks, sources)
        cons2 = block_constraints(
            query, data, max_block_size=2, p_values=P_VALUES, lp_stats=stats
        )
        cons1 = block_constraints(
            query, data, max_block_size=1, p_values=P_VALUES, lp_stats=stats
        )
        f1 = flow_bound(attrs, cons1)
        fs2 = flow_bound(attrs, cons2)
        orders = cycle_candidate_orders(_walk_order(query))
        f2, best_order = flow_bound_over_orders(attrs, cons2, orders)
        f2_walk = flow_bound(attrs, cons2, order=_walk_order(query))
        poly2 = math.nan
        poly_s = math.nan
        if len(attrs) <= POLY_MAX_VARS:
            t0 = time.perf_counter()
            poly2 = 2.0 ** polymatroid_bound(attrs, cons2)
            poly_s = time.perf_counter() - t0
        p = prior[q["query_id"]]
        cover2 = float(p["cover_bs2"])
        ent1 = float(p["entropic"])
        # 5 queries at |V|=10 have no exact count (DuckDB timeout in the
        # suite evaluation); they are kept, with the truth-relative columns
        # left empty, exactly as long_suite_eval.csv reports them.
        true = int(p["true_count"]) if p["true_count"] else None
        rows.append(
            {
                "query_id": q["query_id"],
                "n_vars": len(attrs),
                "n_constraints_bs2": len(cons2),
                "cover_bs1": p["cover_bs1"],
                "entropic_bs1": p["entropic"],
                "flow_bs1": f"{f1.bound:.6e}",
                "cover_bs2": p["cover_bs2"],
                "flow_bs2_simple": f"{fs2.bound:.6e}",
                "flow_bs2_walk": f"{f2_walk.bound:.6e}",
                "flow_bs2": f"{f2.bound:.6e}",
                "poly_bs2": f"{poly2:.6e}",
                "true_count": true if true is not None else "",
                "best_order": "".join(a[1:] for a in best_order) if best_order else "",
                "flow_bs1_eq_entropic": abs(f1.log_bound - math.log2(ent1)) < 1e-6,
                "cover_bs2_over_flow_bs2": f"{cover2 / f2.bound:.4f}",
                "flow_bs2_over_poly_bs2": f"{f2.bound / poly2:.4f}",
                "flow_bs2_walk_over_flow_bs2": f"{f2_walk.bound / f2.bound:.4f}",
                "entropic_bs1_over_flow_bs2": f"{ent1 / f2.bound:.4f}",
                "flow_bs2_over_true": f"{f2.bound / true:.4f}" if true else "",
                "cover_bs2_over_true": f"{cover2 / true:.4f}" if true else "",
                "valid_flow_bs2": f2.bound >= true * (1 - 1e-9) if true else "",
                "flow_bs1_s": f"{f1.solve_seconds:.4f}",
                "flow_bs2_s_per_order": f"{f2.solve_seconds / (len(orders) + 1):.4f}",
                "flow_bs2_s_total": f"{f2.solve_seconds:.4f}",
                "poly_bs2_s": f"{poly_s:.4f}",
                "cover_bs2_lp_s": p["cover_bs2_s"],
                "entropic_s": p["entropic_s"],
                "flow_bs2_vars": f2.n_variables,
            }
        )
        print(
            f"  {q['query_id']:<9} {len(attrs):>3} {cover2:>10.3e} {f2.bound:>10.3e} "
            f"{poly2:>10.3e} {ent1:>10.3e} {true or 0:>9d} {cover2 / f2.bound:>6.2f} "
            f"{f2.bound / poly2:>6.3f} {f2.solve_seconds:>7.3f} {poly_s:>7.2f}",
            flush=True,
        )
    _write("flow_long_suite.csv", rows)

    summary: list[dict[str, object]] = []
    for k in ["all", *sorted({int(str(r["n_vars"])) for r in rows})]:
        rs = [r for r in rows if k == "all" or int(str(r["n_vars"])) == k]

        def med(col: str, rs: list[dict[str, object]] = rs) -> float:
            vals = [float(str(r[col])) for r in rs if str(r[col]) != ""]
            return statistics.median(vals) if vals else math.nan

        summary.append(
            {
                "n_vars": k,
                "n_queries": len(rs),
                "n_flow_bs1_eq_entropic": sum(
                    bool(r["flow_bs1_eq_entropic"]) for r in rs
                ),
                "n_checked": sum(1 for r in rs if r["true_count"] != ""),
                "n_valid_flow_bs2": sum(r["valid_flow_bs2"] is True for r in rs),
                "median_cover_bs2_over_flow_bs2": f"{med('cover_bs2_over_flow_bs2'):.3f}",
                "min_cover_bs2_over_flow_bs2": f"{min(float(str(r['cover_bs2_over_flow_bs2'])) for r in rs):.3f}",
                "max_cover_bs2_over_flow_bs2": f"{max(float(str(r['cover_bs2_over_flow_bs2'])) for r in rs):.3f}",
                "median_flow_bs2_over_poly_bs2": f"{med('flow_bs2_over_poly_bs2'):.4f}",
                "max_flow_bs2_over_poly_bs2": f"{max(float(str(r['flow_bs2_over_poly_bs2'])) for r in rs):.4f}",
                "n_flow_bs2_eq_poly_bs2": sum(
                    float(str(r["flow_bs2_over_poly_bs2"])) < 1 + 1e-6 for r in rs
                ),
                "median_flow_bs2_walk_over_flow_bs2": f"{med('flow_bs2_walk_over_flow_bs2'):.4f}",
                "median_entropic_bs1_over_flow_bs2": f"{med('entropic_bs1_over_flow_bs2'):.1f}",
                "median_flow_bs2_over_true": f"{med('flow_bs2_over_true'):.1f}",
                "median_cover_bs2_over_true": f"{med('cover_bs2_over_true'):.1f}",
                "median_flow_bs2_s_total": f"{med('flow_bs2_s_total'):.3f}",
                "median_flow_bs2_s_per_order": f"{med('flow_bs2_s_per_order'):.4f}",
                "median_poly_bs2_s": f"{med('poly_bs2_s'):.3f}",
                "median_entropic_s": f"{med('entropic_s'):.3f}",
            }
        )
    _write("flow_long_suite_summary.csv", summary)
    for s in summary:
        print("  " + ", ".join(f"{k}={v}" for k, v in s.items()))


def part_scale() -> None:
    n_v, n_e = 100, 400
    ref: dict[int, dict[str, str]] = {}
    scale_csv = ROOT / "results" / "scalability" / "scale_entropic.csv"
    if scale_csv.exists():
        with open(scale_csv) as fh:
            ref = {int(r["k"]): r for r in csv.DictReader(fh)}
    rows: list[dict[str, object]] = []
    print("3. Flow-LP solve time on the cycle sweep")
    print(
        f"  {'k':>3} {'flow1_vars':>10} {'flow1_ms':>9} {'flow2_vars':>10} "
        f"{'flow2_ms/ord':>12} {'flow2_ms_tot':>12} {'poly2_ms':>10} {'f2/p2':>6}"
    )
    for k in COVER_K_VALUES:
        q, data = _build_cycle(k, n_v, n_e)
        attrs = sorted(q.all_attributes)
        blocks2 = enumerate_blocks(q, max_block_size=2)
        stats2 = compute_all_lp_block_extensions(q, blocks2, data, p_values=P_VALUES)
        cons1 = block_constraints(
            q, data, max_block_size=1, p_values=P_VALUES, lp_stats=stats2
        )
        cons2 = block_constraints(
            q, data, max_block_size=2, p_values=P_VALUES, lp_stats=stats2
        )
        # cover LP on the same block statistics (DAG + hyperpath + LP; the
        # statistics are precomputed so this is inference time only)
        t0 = time.perf_counter()
        cover2 = estimate_lph_ceg(
            q, data, max_block_size=2, p_values=P_VALUES, lp_stats=stats2
        ).estimate
        cover2_ms = (time.perf_counter() - t0) * 1e3
        f1 = flow_bound(attrs, cons1)
        f1_ms = statistics.median(
            flow_bound(attrs, cons1).solve_seconds * 1e3 for _ in range(N_TIMING_RUNS)
        )
        orders = cycle_candidate_orders(_walk_order(q))
        f2, _ = flow_bound_over_orders(attrs, cons2, orders)
        per_order_ms = statistics.median(
            flow_bound(attrs, cons2, order=orders[0]).solve_seconds * 1e3
            for _ in range(N_TIMING_RUNS)
        )
        poly2_ms = math.nan
        poly2 = math.nan
        if k <= POLY_MAX_VARS:
            t0 = time.perf_counter()
            poly2 = 2.0 ** polymatroid_bound(attrs, cons2)
            poly2_ms = (time.perf_counter() - t0) * 1e3
        r = ref.get(k, {})
        ent = float(r["entropic_bound"]) if r.get("entropic_bound") else math.nan
        rows.append(
            {
                "k": k,
                "flow_bs1_vars": f1.n_variables,
                "flow_bs1_ms": f"{f1_ms:.2f}",
                "flow_bs1_bound": f"{f1.bound:.6e}",
                "entropic_bound": r.get("entropic_bound", ""),
                "flow_bs1_eq_entropic": (
                    abs(f1.log_bound - math.log2(ent)) < 1e-6
                    if not math.isnan(ent)
                    else ""
                ),
                "cover_ms": r.get("cover_ms", ""),
                "entropic_ms": r.get("entropic_ms", ""),
                "cover_bs2_bound": f"{cover2:.6e}",
                "cover_bs2_ms": f"{cover2_ms:.2f}",
                "flow_bs2_vars": f2.n_variables,
                "n_orders": len(orders),
                "flow_bs2_ms_per_order": f"{per_order_ms:.2f}",
                "flow_bs2_ms_total": f"{f2.solve_seconds * 1e3:.1f}",
                "flow_bs2_bound": f"{f2.bound:.6e}",
                "poly_bs2_bound": f"{poly2:.6e}",
                "poly_bs2_ms": f"{poly2_ms:.1f}",
                "flow_bs2_over_poly_bs2": f"{f2.bound / poly2:.4f}",
                "cover_bs2_over_flow_bs2": f"{cover2 / f2.bound:.4f}",
            }
        )
        print(
            f"  {k:>3} {f1.n_variables:>10} {f1_ms:>9.2f} {f2.n_variables:>10} "
            f"{per_order_ms:>12.2f} {f2.solve_seconds * 1e3:>12.1f} "
            f"{poly2_ms:>10.1f} {f2.bound / poly2:>6.3f}",
            flush=True,
        )
    _write("flow_scale.csv", rows)


def _write(name: str, rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {OUT_DIR / name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("2x2", "suite", "scale", "all"), default="all")
    args = ap.parse_args()
    if args.part in ("2x2", "all"):
        part_2x2()
    if args.part in ("scale", "all"):
        part_scale()
    if args.part in ("suite", "all"):
        part_long_suite()


if __name__ == "__main__":
    main()
