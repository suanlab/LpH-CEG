"""Collection cost at five times the films, through the sparse-product path.

``evaluate_imdb_suite.py --data-dir data/imdb_cycles_n1000`` collected the
bs<=2 catalog of the 40 |V|=6 IMDb cycles at 1,000 films by materializing
the pair joins (median 48 s per query, up to 25 min and 16 GiB).  This
collects the same blocks and interfaces with ``compute_sparse_block_extensions``
(equal statistics, tests/unit/test_pushdown_stats.py) but with eleven norms
where the materialized run used three, one query per fresh process,
and times the statistics, the constraint build and the polymatroid solve
(eleven norms), with the process's peak RSS.  The same is done at 200
films for the growth ratio.

Output: results/imdb/imdb_scale_sparse.csv and _summary.csv
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import resource
import statistics
import time
from pathlib import Path

import scripts.evaluate_imdb_suite as ev
from scripts.pset_fair import P11, SUITES
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import block_constraints, polymatroid_bound
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.statistics.pushdown_stats import compute_sparse_block_extensions

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "imdb"
WORKERS = 8
DATA = {
    "n200": ROOT / "data" / "imdb_cycles",
    "n1000": ROOT / "data" / "imdb_cycles_n1000",
}


def evaluate(item: tuple[str, dict]) -> dict[str, object]:  # type: ignore[type-arg]
    tag, q = item
    ev.DATA = DATA[tag]
    query, data = ev.build_imdb_query(q)
    attrs = sorted(query.all_attributes)
    blocks = enumerate_blocks(query, max_block_size=2)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    t0 = time.perf_counter()
    st = dict(
        compute_all_lp_block_extensions(
            query, [b for b in blocks if len(b.relations) == 1], data, p_values=P11
        )
    )
    for b in blocks:
        if len(b.relations) == 2:
            st.update(compute_sparse_block_extensions(query, b, data, p_values=P11))
    t1 = time.perf_counter()
    cons = block_constraints(query, data, max_block_size=2, p_values=P11, lp_stats=st)
    log2 = polymatroid_bound(attrs, cons)
    t2 = time.perf_counter()
    return {
        "data": tag,
        "query_id": q["query_id"],
        "rows": sum(len(r) for _, r in data.values()),
        "collect_s": f"{t1 - t0:.3f}",
        "constraints_and_solve_s": f"{t2 - t1:.3f}",
        "log2_poly_bs2": f"{log2:.6f}",
        "rss_loaded_mb": f"{rss0:.0f}",
        "rss_peak_mb": f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}",
    }


def main() -> None:
    qs = [
        q
        for q in json.loads(SUITES["imdb"][0].read_text())["queries"]
        if q["n_vars"] == 6
    ]
    items = [(tag, q) for tag in DATA for q in qs]
    with multiprocessing.Pool(processes=WORKERS, maxtasksperchild=1) as pool:
        rows = pool.map(evaluate, items, chunksize=1)
    with open(OUT / "imdb_scale_sparse.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for tag in DATA:
        rs = [r for r in rows if r["data"] == tag]
        c = [float(str(r["collect_s"])) for r in rs]
        summary.append(
            {
                "data": tag,
                "n": len(rs),
                "median_rows": statistics.median(int(str(r["rows"])) for r in rs),
                "median_collect_s": f"{statistics.median(c):.3f}",
                "max_collect_s": f"{max(c):.3f}",
                "median_solve_s": f"{statistics.median(float(str(r['constraints_and_solve_s'])) for r in rs):.3f}",
                "max_peak_mb": max(int(str(r["rss_peak_mb"])) for r in rs),
                "median_peak_over_loaded_mb": statistics.median(
                    int(str(r["rss_peak_mb"])) - int(str(r["rss_loaded_mb"]))
                    for r in rs
                ),
            }
        )
    with open(OUT / "imdb_scale_sparse_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
