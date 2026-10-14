"""Does the entropic baseline gain from more lp-norms than {1, 2, inf}?

The paper's LpBound reimplementation (the entropic bound on single-relation
statistics) uses P = {1, 2, inf}, as the cover LP does.  The published
LpBound can use more norms.  This recomputes the bs=1 entropic bound -- via
the flow LP, which equals it (verified 240/240) -- with P = {1, ..., 10, inf}
on both long suites (Hetionet |V|=6/8/10, IMDb |V|=6/8/10) and reports how
much tighter it gets.  If the gain is small, the baseline was not weakened by
its p-set.

Output: results/suite/lpbound_pset.csv and lpbound_pset_summary.csv
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from scripts.evaluate_imdb_suite import build_imdb_query
from scripts.evaluate_query_suite import build_hetionet_query
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import block_constraints, flow_bound
from src.statistics.lp_stats import compute_all_lp_block_extensions

ROOT = Path(__file__).resolve().parent.parent
P3 = (1.0, 2.0, math.inf)
P11 = (*(float(p) for p in range(1, 11)), math.inf)


def bounds(query, data) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    attrs = sorted(query.all_attributes)
    stats = compute_all_lp_block_extensions(
        query, enumerate_blocks(query, max_block_size=1), data, p_values=P11
    )
    out = []
    for ps in (P3, P11):
        cons = block_constraints(
            query, data, max_block_size=1, p_values=ps, lp_stats=stats
        )
        out.append(flow_bound(attrs, cons).bound)
    return out[0], out[1]


def main() -> None:
    rows: list[dict[str, object]] = []
    suites = (
        (
            "hetionet",
            ROOT / "benchmarks" / "query_suite_v2_long.json",
            build_hetionet_query,
        ),
        ("imdb", ROOT / "benchmarks" / "query_suite_v3_imdb.json", build_imdb_query),
    )
    for name, path, build in suites:
        for q in json.loads(path.read_text())["queries"]:
            built = build(q)
            if built is None:
                continue
            b3, b11 = bounds(*built)
            rows.append(
                {
                    "suite": name,
                    "query_id": q["query_id"],
                    "n_vars": q["n_vars"],
                    "entropic_p3": f"{b3:.6e}",
                    "entropic_p11": f"{b11:.6e}",
                    "p3_over_p11": f"{b3 / b11:.6f}",
                }
            )
            print(name, q["query_id"], rows[-1]["p3_over_p11"], flush=True)
    out = ROOT / "results" / "suite"
    with open(out / "lpbound_pset.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for name in ("hetionet", "imdb", "all"):
        v = [float(str(r["p3_over_p11"])) for r in rows if name in ("all", r["suite"])]
        summary.append(
            {
                "suite": name,
                "n": len(v),
                "median_p3_over_p11": f"{statistics.median(v):.6f}",
                "max_p3_over_p11": f"{max(v):.6f}",
                "n_gain_over_1pct": sum(x > 1.01 for x in v),
            }
        )
    with open(out / "lpbound_pset_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
