"""CorrBound proxy: the entropic program plus pairwise join sizes.

CorrBound (Mayer et al., SIGMOD 2026) adds inner products of degree vectors
across join columns to LpBound's entropic program.  For two relations
``R(a,b), S(b,c)`` that inner product is ``sum_b deg_R(b) deg_S(b) =
|R join S|``, i.e. the size of the pair join.  As a constraint it is
``h(abc) <= log |R join S|`` -- empty conditioning, hence *simple* -- so the
flow LP over single-relation lp statistics plus these sizes equals the
polymatroid bound exactly (Im et al. 2025).  That is the strongest program
on CorrBound's two-relation statistic; we do not reproduce CorrBound's
sketches, low-rank approximations or multi-relation inner products, so this
is a proxy for its *statistic*, not a reimplementation of the system.

Variants, all with the flow LP as the program:

* ``flow_bs1``        -- single-relation lp statistics (LpBound's LPflow);
* ``flow_bs1_sizes``  -- plus every pair-join size (CorrBound proxy);
* ``flow_bs2``        -- the full bs=2 catalogue (interface-conditioned).

Outputs ``results/pushdown/corrbound_proxy_{2x2,scale,long_suite}.csv`` and
``corrbound_proxy_summary.csv``.

Usage:
    PYTHONPATH=. python scripts/corrbound_proxy.py [--part 2x2|scale|suite|all]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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
from src.statistics.lp_stats import compute_all_lp_block_extensions

if TYPE_CHECKING:
    from src.statistics.block_stats import RelationDataMap
    from src.utils.types import QueryHypergraph

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "pushdown"


def _bounds(
    query: QueryHypergraph,
    data: RelationDataMap,
    stats: dict,  # type: ignore[type-arg]
    orders: list[tuple[str, ...]],
) -> dict[str, float]:
    attrs = sorted(query.all_attributes)
    cons1 = block_constraints(
        query, data, max_block_size=1, p_values=P_VALUES, lp_stats=stats
    )
    cons2 = block_constraints(
        query, data, max_block_size=2, p_values=P_VALUES, lp_stats=stats
    )
    known = set(cons1)
    # pair-join sizes: block constraints with empty conditioning (one per
    # block; every p gives the same value, so keep p = 1)
    sizes = [c for c in cons2 if c not in known and not c.conditioning and c.p == 1]
    b1 = flow_bound(attrs, cons1).bound
    bsz = flow_bound(attrs, cons1 + sizes).bound
    b2 = flow_bound_over_orders(attrs, cons2, orders)[0].bound
    return {
        "flow_bs1": b1,
        "flow_bs1_sizes": bsz,
        "flow_bs2": b2,
        "n_sizes": len(sizes),
    }


def _row(
    prefix: dict[str, object], b: dict[str, float], true: int | None
) -> dict[str, object]:
    r: dict[str, object] = dict(prefix)
    r.update(
        {
            "n_sizes": int(b["n_sizes"]),
            "flow_bs1": f"{b['flow_bs1']:.6e}",
            "flow_bs1_sizes": f"{b['flow_bs1_sizes']:.6e}",
            "flow_bs2": f"{b['flow_bs2']:.6e}",
            "bs1_over_sizes": f"{b['flow_bs1'] / b['flow_bs1_sizes']:.4f}",
            "sizes_over_full": f"{b['flow_bs1_sizes'] / b['flow_bs2']:.4f}",
            "true_count": true if true is not None else "",
            "sizes_below_true": int(
                true is not None and b["flow_bs1_sizes"] < true * (1 - 1e-9)
            ),
        }
    )
    return r


def part_2x2() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fam in FAMILIES:
        for seed in SEEDS:
            q, d = sh.build(sh.FAMILIES[fam], seed=seed)
            stats = compute_all_lp_block_extensions(
                q, enumerate_blocks(q, max_block_size=2), d, p_values=P_VALUES
            )
            b = _bounds(q, d, stats, _orders(q))
            rows.append(_row({"family": fam, "seed": seed}, b, None))
            print(fam, seed, rows[-1]["bs1_over_sizes"], rows[-1]["sizes_over_full"])
    _write("corrbound_proxy_2x2.csv", rows)
    return rows


def part_scale() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k in COVER_K_VALUES:
        q, d = _build_cycle(k, 100, 400)
        stats = compute_all_lp_block_extensions(
            q, enumerate_blocks(q, max_block_size=2), d, p_values=P_VALUES
        )
        b = _bounds(q, d, stats, cycle_candidate_orders(_walk_order(q)))
        rows.append(_row({"k": k}, b, None))
        print("cycle", k, rows[-1]["bs1_over_sizes"], rows[-1]["sizes_over_full"])
    _write("corrbound_proxy_scale.csv", rows)
    return rows


def part_long_suite() -> list[dict[str, object]]:
    with open(SUITE) as fh:
        suite = json.load(fh)
    with open(EVAL) as fh:
        prior = {r["query_id"]: r for r in csv.DictReader(fh)}
    cat = StatsCatalogue(P_VALUES)
    rows: list[dict[str, object]] = []
    for q in suite["queries"]:
        built = build_hetionet_query(q)
        if built is None:
            continue
        query, data = built
        blocks = enumerate_blocks(query, max_block_size=2)
        stats, _, _ = cat.stats_for(query, data, blocks, relation_sources(q))
        b = _bounds(query, data, stats, cycle_candidate_orders(_walk_order(query)))
        p = prior[q["query_id"]]
        true = int(p["true_count"]) if p["true_count"] else None
        prefix = {"query_id": q["query_id"], "n_vars": len(query.all_attributes)}
        rows.append(_row(prefix, b, true))
        print(
            q["query_id"],
            rows[-1]["bs1_over_sizes"],
            rows[-1]["sizes_over_full"],
            flush=True,
        )
    _write("corrbound_proxy_long_suite.csv", rows)
    summary: list[dict[str, object]] = []
    for nv in ("all", 6, 8, 10):
        rs = rows if nv == "all" else [r for r in rows if r["n_vars"] == nv]
        a = [float(str(r["bs1_over_sizes"])) for r in rs]
        f = [float(str(r["sizes_over_full"])) for r in rs]
        summary.append(
            {
                "n_vars": nv,
                "n_queries": len(rs),
                "median_bs1_over_sizes": f"{statistics.median(a):.4f}",
                "min_bs1_over_sizes": f"{min(a):.4f}",
                "max_bs1_over_sizes": f"{max(a):.4f}",
                "median_sizes_over_full": f"{statistics.median(f):.4f}",
                "min_sizes_over_full": f"{min(f):.4f}",
                "max_sizes_over_full": f"{max(f):.4f}",
                "n_full_below_sizes": sum(x > 1.0001 for x in f),
                "n_checked": sum(r["true_count"] != "" for r in rs),
                "n_sizes_below_true": sum(int(str(r["sizes_below_true"])) for r in rs),
            }
        )
    _write("corrbound_proxy_summary.csv", summary)
    return rows


def _write(name: str, rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_DIR / name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", default="all", choices=["2x2", "scale", "suite", "all"])
    part = ap.parse_args().part
    if part in ("2x2", "all"):
        part_2x2()
    if part in ("scale", "all"):
        part_scale()
    if part in ("suite", "all"):
        part_long_suite()


if __name__ == "__main__":
    main()
