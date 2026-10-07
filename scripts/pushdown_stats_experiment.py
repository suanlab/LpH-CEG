"""B4: which block statistics need the materialized join, and do they matter?

For every two-relation block the interfaces split into *join-free* ones
(computable from single-relation degree tables in ``O(|R|+|S|)``,
:mod:`src.statistics.pushdown_stats`) and *join-required* ones (common
neighbour counts; ``{a, c}`` on a binary block ``R(a,b), S(b,c)``).  This
script measures, with the flow LP as the inference engine:

* ``flow_bs1``            -- single-relation statistics (LpBound's LPflow);
* ``flow_bs2_free``       -- bs=1 plus the join-free bs=2 interfaces;
* ``flow_bs2_join``       -- bs=1 plus the join-required bs=2 interfaces;
* ``flow_bs2``            -- the full bs=2 catalogue;

and the cost of the join-free statistics against the materialized block
join, on the synthetic families, the cycle sweep and the 120-query long
suite.  Outputs ``results/pushdown/pushdown_{2x2,scale,long_suite}.csv``
and ``pushdown_long_suite_summary.csv``.

Usage:
    PYTHONPATH=. python scripts/pushdown_stats_experiment.py [--part 2x2|scale|suite|all]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING

from scripts import statistics_hierarchy as sh
from scripts.evaluate_query_suite import P_VALUES, build_hetionet_query
from scripts.flow_bound_blocks import FAMILIES, SEEDS, _orders, _walk_order
from scripts.scalability_vs_entropic import COVER_K_VALUES, _build_cycle
from scripts.stats_catalogue import EVAL, SUITE, StatsCatalogue, relation_sources
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import (
    block_constraints,
    cycle_candidate_orders,
    flow_bound,
    flow_bound_over_orders,
)
from src.statistics.lp_stats import (
    LpBlockExtensionStats,
    compute_all_lp_block_extensions,
)
from src.statistics.pushdown_stats import (
    codegree_interface,
    codegree_l2_upper_bound,
    compute_pushdown_block_extensions,
    join_free_interfaces,
    sketch_codegree_l2,
)

if TYPE_CHECKING:
    from src.statistics.block_stats import RelationDataMap
    from src.utils.types import Block, Interface, QueryHypergraph

ROOT = Path(__file__).resolve().parent.parent
SKETCH_PROBES = 64
OUT_DIR = ROOT / "results" / "pushdown"
StatsMap = dict[tuple["Block", "Interface"], "LpBlockExtensionStats"]


def split_stats(
    query: QueryHypergraph, stats: StatsMap
) -> tuple[StatsMap, StatsMap, int, int]:
    """Split a bs<=2 catalogue into (bs1 + join-free bs2, bs1 + join-required bs2)."""
    free: StatsMap = {}
    join: StatsMap = {}
    n_free = n_join = 0
    kinds_cache: dict[Block, dict[Interface, str | None]] = {}
    for (block, iface), st in stats.items():
        if len(block.relations) == 1:
            free[(block, iface)] = st
            join[(block, iface)] = st
            continue
        if block not in kinds_cache:
            kinds_cache[block] = join_free_interfaces(query, block)
        if kinds_cache[block][iface] is None:
            join[(block, iface)] = st
            n_join += 1
        else:
            free[(block, iface)] = st
            n_free += 1
    return free, join, n_free, n_join


def _l2_only(value: float) -> LpBlockExtensionStats:
    """A stats entry carrying only an l2 norm (other p skipped by the LP)."""
    return LpBlockExtensionStats(
        degree_sequence=(),
        lp_norms={2.0: value},
        interface_size=2,
        count=0,
        max_extension=0,
        avg_extension=0.0,
        log_max_extension=0.0,
    )


def codegree_surrogates(
    query: QueryHypergraph,
    blocks: tuple[Block, ...],
    data: RelationDataMap,
    free: StatsMap,
) -> tuple[StatsMap, StatsMap, float, float, float]:
    """Join-free maps with the codegree l2 replaced by the triangle bound or a
    sketch; returns (tri_map, sketch_map, tri_s, sketch_s, max_rel_se)."""
    tri: StatsMap = dict(free)
    sk: StatsMap = dict(free)
    tri_s = sk_s = 0.0
    max_se = 0.0
    for block in blocks:
        if len(block.relations) != 2:
            continue
        iface = codegree_interface(query, block)
        t0 = time.perf_counter()
        tri[(block, iface)] = _l2_only(codegree_l2_upper_bound(query, block, data))
        t1 = time.perf_counter()
        est, se = sketch_codegree_l2(query, block, data, n_probes=SKETCH_PROBES)
        sk[(block, iface)] = _l2_only(est)
        t2 = time.perf_counter()
        tri_s += t1 - t0
        sk_s += t2 - t1
        max_se = max(max_se, se)
    return tri, sk, tri_s, sk_s, max_se


def pushdown_all(
    query: QueryHypergraph, blocks: tuple[Block, ...], data: RelationDataMap
) -> tuple[StatsMap, float]:
    """Join-free statistics for every block, and the seconds they took."""
    t0 = time.perf_counter()
    out: StatsMap = {}
    for block in blocks:
        out.update(
            compute_pushdown_block_extensions(query, block, data, p_values=P_VALUES)
        )
    return out, time.perf_counter() - t0


def check_pushdown(full: StatsMap, pushed: StatsMap) -> int:
    """Number of join-free interfaces whose pushdown norms differ from the join."""
    bad = 0
    for key, st in pushed.items():
        ref = full[key]
        for p in P_VALUES:
            if abs(st.norm(p) - ref.norm(p)) > 1e-9 * max(1.0, ref.norm(p)):
                bad += 1
                break
        else:
            if st.count != ref.count or st.max_extension != ref.max_extension:
                bad += 1
    return bad


def _bounds(
    query: QueryHypergraph,
    data: RelationDataMap,
    stats: StatsMap,
    orders: list[tuple[str, ...]],
) -> dict[str, float]:
    attrs = sorted(query.all_attributes)
    free, join, n_free, n_join = split_stats(query, stats)
    out: dict[str, float] = {"n_free": n_free, "n_join": n_join}
    for name, st in (("bs2", stats), ("bs2_free", free), ("bs2_join", join)):
        cons = block_constraints(
            query, data, max_block_size=2, p_values=P_VALUES, lp_stats=st
        )
        res, _ = flow_bound_over_orders(attrs, cons, orders)
        out[f"flow_{name}"] = res.bound
        out[f"flow_{name}_s"] = res.solve_seconds
    cons1 = block_constraints(
        query, data, max_block_size=1, p_values=P_VALUES, lp_stats=stats
    )
    out["flow_bs1"] = flow_bound(attrs, cons1).bound
    # p-ablation of the join-required constraints: which norm of the
    # common-neighbour sequence carries the gain?
    cons_full = block_constraints(
        query, data, max_block_size=2, p_values=P_VALUES, lp_stats=stats
    )
    cons_free = block_constraints(
        query, data, max_block_size=2, p_values=P_VALUES, lp_stats=free
    )
    free_set = set(cons_free)
    join_only = [c for c in cons_full if c not in free_set]
    for name, base, ps in (
        ("bs2_free_l2join", cons_free, (2.0,)),
        ("bs2_free_linfjoin", cons_free, (math.inf,)),
        ("bs1_l2join", cons1, (2.0,)),
    ):
        cons = base + [c for c in join_only if c.p in ps]
        out[f"flow_{name}"] = flow_bound_over_orders(attrs, cons, orders)[0].bound
    # linear-time surrogates for the codegree l2 (valid bound / sketch)
    blocks = tuple(sorted({b for b, _ in stats}, key=lambda b: sorted(b.relations)))
    tri, sk, tri_s, sk_s, max_se = codegree_surrogates(query, blocks, data, free)
    for name, st in (("bs2_tri", tri), ("bs2_sketch", sk)):
        cons = block_constraints(
            query, data, max_block_size=2, p_values=P_VALUES, lp_stats=st
        )
        out[f"flow_{name}"] = flow_bound_over_orders(attrs, cons, orders)[0].bound
    out["tri_s"] = tri_s
    out["sketch_s"] = sk_s
    out["sketch_max_rel_se"] = max_se
    return out


def _row(
    prefix: dict[str, object],
    b: dict[str, float],
    join_s: float,
    push_s: float,
    n_bad: int,
) -> dict[str, object]:
    r: dict[str, object] = dict(prefix)
    r.update(
        {
            "n_free_ifaces": int(b["n_free"]),
            "n_join_ifaces": int(b["n_join"]),
            "flow_bs1": f"{b['flow_bs1']:.6e}",
            "flow_bs2_free": f"{b['flow_bs2_free']:.6e}",
            "flow_bs2_join": f"{b['flow_bs2_join']:.6e}",
            "flow_bs2": f"{b['flow_bs2']:.6e}",
            "free_over_full": f"{b['flow_bs2_free'] / b['flow_bs2']:.4f}",
            "join_over_full": f"{b['flow_bs2_join'] / b['flow_bs2']:.4f}",
            "bs1_over_full": f"{b['flow_bs1'] / b['flow_bs2']:.4f}",
            "free_l2join_over_full": f"{b['flow_bs2_free_l2join'] / b['flow_bs2']:.4f}",
            "free_linfjoin_over_full": f"{b['flow_bs2_free_linfjoin'] / b['flow_bs2']:.4f}",
            "bs1_l2join_over_full": f"{b['flow_bs1_l2join'] / b['flow_bs2']:.4f}",
            "flow_bs2_tri": f"{b['flow_bs2_tri']:.6e}",
            "flow_bs2_sketch": f"{b['flow_bs2_sketch']:.6e}",
            "tri_over_full": f"{b['flow_bs2_tri'] / b['flow_bs2']:.4f}",
            "sketch_over_full": f"{b['flow_bs2_sketch'] / b['flow_bs2']:.4f}",
            "tri_gain_share": _share(b["flow_bs1"], b["flow_bs2_tri"], b["flow_bs2"]),
            "tri_s": f"{b['tri_s']:.4f}",
            "sketch_s": f"{b['sketch_s']:.4f}",
            "sketch_max_rel_se": f"{b['sketch_max_rel_se']:.4f}",
            # share of the log-gain bs1 -> bs2 that the join-free statistics
            # alone realise (1 = all of it, 0 = none)
            "free_gain_share": _share(b["flow_bs1"], b["flow_bs2_free"], b["flow_bs2"]),
            "join_gain_share": _share(b["flow_bs1"], b["flow_bs2_join"], b["flow_bs2"]),
            "join_build_s": f"{join_s:.4f}",
            "pushdown_s": f"{push_s:.4f}",
            "join_over_pushdown_cost": f"{join_s / push_s:.2f}" if push_s > 0 else "",
            "n_pushdown_mismatch": n_bad,
        }
    )
    return r


def _share(b1: float, part: float, full: float) -> str:
    denom = math.log(b1) - math.log(full)
    if denom <= 1e-12:
        return ""
    return f"{(math.log(b1) - math.log(part)) / denom:.4f}"


def _timed_full(
    query: QueryHypergraph, blocks: tuple[Block, ...], data: RelationDataMap
) -> tuple[StatsMap, float]:
    t0 = time.perf_counter()
    st = compute_all_lp_block_extensions(query, blocks, data, p_values=P_VALUES)
    return st, time.perf_counter() - t0


def _print_header() -> None:
    print(
        f"  {'case':<14} {'|V|':>3} {'flow1':>11} {'free2':>11} {'join2':>11} "
        f"{'full2':>11} {'free/full':>9} {'join/full':>9} {'fr+l2/fu':>9} {'b1+l2/fu':>9} "
        f"{'join_s':>7} {'push_s':>7}"
    )


def _print(case: str, nv: int, b: dict[str, float], js: float, ps: float) -> None:
    print(
        f"  {case:<14} {nv:>3} {b['flow_bs1']:>11.4g} {b['flow_bs2_free']:>11.4g} "
        f"{b['flow_bs2_join']:>11.4g} {b['flow_bs2']:>11.4g} "
        f"{b['flow_bs2_free'] / b['flow_bs2']:>9.3f} "
        f"{b['flow_bs2_join'] / b['flow_bs2']:>9.3f} "
        f"{b['flow_bs2_free_l2join'] / b['flow_bs2']:>9.3f} "
        f"{b['flow_bs1_l2join'] / b['flow_bs2']:>9.3f} "
        f"{b['flow_bs2_tri'] / b['flow_bs2']:>7.3f} "
        f"{b['flow_bs2_sketch'] / b['flow_bs2']:>7.3f} "
        f"{js:>7.3f} {ps:>7.3f}",
        flush=True,
    )


def part_2x2() -> None:
    rows: list[dict[str, object]] = []
    print("1. Families: join-free vs join-required block statistics (flow LP)")
    _print_header()
    for fam in FAMILIES:
        for seed in SEEDS:
            q, d = sh.build(sh.FAMILIES[fam], seed=seed)
            blocks = enumerate_blocks(q, max_block_size=2)
            stats, join_s = _timed_full(q, blocks, d)
            pushed, push_s = pushdown_all(q, blocks, d)
            n_bad = check_pushdown(stats, pushed)
            b = _bounds(q, d, stats, _orders(q))
            rows.append(
                _row(
                    {"family": fam, "seed": seed, "n_vars": len(q.all_attributes)},
                    b,
                    join_s,
                    push_s,
                    n_bad,
                )
            )
            _print(f"{fam}/{seed}", len(q.all_attributes), b, join_s, push_s)
    _write("pushdown_2x2.csv", rows)


def part_scale() -> None:
    rows: list[dict[str, object]] = []
    print("2. Cycle sweep")
    _print_header()
    for k in COVER_K_VALUES:
        q, d = _build_cycle(k, 100, 400)
        blocks = enumerate_blocks(q, max_block_size=2)
        stats, join_s = _timed_full(q, blocks, d)
        pushed, push_s = pushdown_all(q, blocks, d)
        n_bad = check_pushdown(stats, pushed)
        b = _bounds(q, d, stats, cycle_candidate_orders(_walk_order(q)))
        rows.append(_row({"k": k, "n_vars": k}, b, join_s, push_s, n_bad))
        _print(f"cycle-{k}", k, b, join_s, push_s)
    _write("pushdown_scale.csv", rows)


def part_long_suite() -> None:
    with open(SUITE) as fh:
        suite = json.load(fh)
    with open(EVAL) as fh:
        prior = {r["query_id"]: r for r in csv.DictReader(fh)}
    cat = StatsCatalogue(P_VALUES)
    rows: list[dict[str, object]] = []
    print("3. Long-cycle suite")
    _print_header()
    for q in suite["queries"]:
        built = build_hetionet_query(q)
        if built is None:
            continue
        query, data = built
        blocks = enumerate_blocks(query, max_block_size=2)
        before = cat.build_seconds
        stats, misses, _ = cat.stats_for(query, data, blocks, relation_sources(q))
        join_s = cat.build_seconds - before  # only the blocks built for this query
        pushed, push_s = pushdown_all(query, blocks, data)
        n_bad = check_pushdown(stats, pushed)
        b = _bounds(query, data, stats, cycle_candidate_orders(_walk_order(query)))
        p = prior[q["query_id"]]
        true = int(p["true_count"]) if p["true_count"] else None
        prefix: dict[str, object] = {
            "query_id": q["query_id"],
            "n_vars": len(query.all_attributes),
            "n_blocks_built": misses,
            "true_count": true if true is not None else "",
            "free_over_true": f"{b['flow_bs2_free'] / true:.4f}" if true else "",
            "full_over_true": f"{b['flow_bs2'] / true:.4f}" if true else "",
            "tri_over_true": f"{b['flow_bs2_tri'] / true:.4f}" if true else "",
            "sketch_over_true": f"{b['flow_bs2_sketch'] / true:.4f}" if true else "",
        }
        rows.append(_row(prefix, b, join_s, push_s, n_bad))
        _print(q["query_id"], len(query.all_attributes), b, join_s, push_s)
    _write("pushdown_long_suite.csv", rows)

    def med(rs: list[dict[str, object]], col: str) -> float:
        vals = [float(str(r[col])) for r in rs if r[col] != ""]
        return statistics.median(vals) if vals else math.nan

    summary: list[dict[str, object]] = []
    for nv in ("all", 6, 8, 10):
        rs = rows if nv == "all" else [r for r in rows if r["n_vars"] == nv]
        summary.append(
            {
                "n_vars": nv,
                "n_queries": len(rs),
                "n_pushdown_mismatch": sum(
                    int(str(r["n_pushdown_mismatch"])) for r in rs
                ),
                "median_free_over_full": f"{med(rs, 'free_over_full'):.4f}",
                "max_free_over_full": f"{max(float(str(r['free_over_full'])) for r in rs):.4f}",
                "n_free_eq_full": sum(
                    float(str(r["free_over_full"])) < 1.0001 for r in rs
                ),
                "median_join_over_full": f"{med(rs, 'join_over_full'):.4f}",
                "n_join_eq_full": sum(
                    float(str(r["join_over_full"])) < 1.0001 for r in rs
                ),
                "n_free_l2join_eq_full": sum(
                    float(str(r["free_l2join_over_full"])) < 1.0001 for r in rs
                ),
                "median_free_l2join_over_full": f"{med(rs, 'free_l2join_over_full'):.4f}",
                "max_free_l2join_over_full": f"{max(float(str(r['free_l2join_over_full'])) for r in rs):.4f}",
                "median_free_linfjoin_over_full": f"{med(rs, 'free_linfjoin_over_full'):.4f}",
                "median_bs1_l2join_over_full": f"{med(rs, 'bs1_l2join_over_full'):.4f}",
                "n_bs1_l2join_eq_full": sum(
                    float(str(r["bs1_l2join_over_full"])) < 1.0001 for r in rs
                ),
                "median_bs1_over_full": f"{med(rs, 'bs1_over_full'):.4f}",
                "median_free_gain_share": f"{med(rs, 'free_gain_share'):.4f}",
                "median_join_gain_share": f"{med(rs, 'join_gain_share'):.4f}",
                "median_tri_over_full": f"{med(rs, 'tri_over_full'):.4f}",
                "max_tri_over_full": f"{max(float(str(r['tri_over_full'])) for r in rs):.4f}",
                "median_tri_gain_share": f"{med(rs, 'tri_gain_share'):.4f}",
                "median_sketch_over_full": f"{med(rs, 'sketch_over_full'):.4f}",
                "min_sketch_over_full": f"{min(float(str(r['sketch_over_full'])) for r in rs):.4f}",
                "max_sketch_over_full": f"{max(float(str(r['sketch_over_full'])) for r in rs):.4f}",
                "n_sketch_below_true": sum(
                    r["sketch_over_true"] != ""
                    and float(str(r["sketch_over_true"])) < 1.0
                    for r in rs
                ),
                "n_tri_below_true": sum(
                    r["tri_over_true"] != "" and float(str(r["tri_over_true"])) < 1.0
                    for r in rs
                ),
                "total_tri_s": f"{sum(float(str(r['tri_s'])) for r in rs):.1f}",
                "total_sketch_s": f"{sum(float(str(r['sketch_s'])) for r in rs):.1f}",
                "median_free_over_true": f"{med(rs, 'free_over_true'):.2f}",
                "median_full_over_true": f"{med(rs, 'full_over_true'):.2f}",
                "total_join_build_s": f"{sum(float(str(r['join_build_s'])) for r in rs):.1f}",
                "total_pushdown_s": f"{sum(float(str(r['pushdown_s'])) for r in rs):.1f}",
                "median_pushdown_s": f"{med(rs, 'pushdown_s'):.4f}",
            }
        )
        print("  " + ", ".join(f"{k}={v}" for k, v in summary[-1].items()))
    _write("pushdown_long_suite_summary.csv", summary)


def _write(name: str, rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {OUT_DIR / name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("2x2", "scale", "suite", "all"), default="all")
    args = ap.parse_args()
    if args.part in ("2x2", "all"):
        part_2x2()
    if args.part in ("scale", "all"):
        part_scale()
    if args.part in ("suite", "all"):
        part_long_suite()


if __name__ == "__main__":
    main()
