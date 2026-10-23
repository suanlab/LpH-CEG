"""Cost against tightness, each catalog in its own process, one implementation.

``cost_accuracy.py`` collects the catalogs of a query in one process, in a
fixed order, and its full catalog materializes the pair joins while
join-free+l2 takes the common-neighbour l2 from a sparse product.  This
repeats the comparison so that the cost differences are not those of the
implementation or of the order:

  lpbound      single-relation statistics (LpBound's input)
  free         + the join-free interfaces of every pair block
  free_l2      + the exact common-neighbour l2 of each block
  sparse_full  + every common-neighbour norm, from the same sparse product
               (with ``free``: the full bs<=2 catalog, no join tuples)
  full         the full bs<=2 catalog from the materialized pair joins

Same queries (Hetionet and IMDb long suites), eleven norms, the 2^|V|
polymatroid program.  Every (query, catalog) pair is one task in a fresh
worker process (``maxtasksperchild=1``); tasks run in a shuffled order.  A
task times its statistics collection (cold, data load excluded), the
constraint build and the polymatroid solve, and reports the process's peak
RSS after loading the data and at the end.  LpBound's flow LP is timed in
the ``lpbound`` task.  ``sparse_full`` must reproduce ``full``'s bound.

Output: results/cost/cost_accuracy_indep.csv and _summary.csv
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import random
import resource
import signal
import statistics
import time
from pathlib import Path

from scripts.cost_accuracy import TRUTH
from scripts.pset_fair import P11, SUITES
from scripts.pushdown_stats_experiment import _l2_only
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import block_constraints, flow_bound, polymatroid_bound
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.statistics.pushdown_stats import (
    codegree_block_extension,
    codegree_interface,
    compute_pushdown_block_extensions,
    compute_sparse_block_extensions,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cost"
WORKERS = 8
TASK_TIMEOUT_S = 3600
SEED = 2026
CATALOGUES = ("lpbound", "free", "free_l2", "sparse_full", "full")


class _TaskTimeoutError(Exception):
    pass


def _alarm(signum, frame):  # type: ignore[no-untyped-def]
    raise _TaskTimeoutError


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def collect(name: str, query, data, singles, pairs) -> dict:  # type: ignore[no-untyped-def,type-arg]
    st = dict(compute_all_lp_block_extensions(query, singles, data, p_values=P11))
    if name == "lpbound":
        return st
    if name == "full":
        st.update(compute_all_lp_block_extensions(query, pairs, data, p_values=P11))
        return st
    for b in pairs:
        if name == "sparse_full":
            st.update(compute_sparse_block_extensions(query, b, data, p_values=P11))
        else:
            st.update(compute_pushdown_block_extensions(query, b, data, p_values=P11))
            if name == "free_l2":
                cd = codegree_block_extension(query, b, data, p_values=(2.0,))
                st[(b, codegree_interface(query, b))] = _l2_only(cd.norm(2.0))
    return st


def evaluate(item: tuple[str, dict, str]) -> dict[str, object]:  # type: ignore[type-arg]
    suite, q, name = item
    row: dict[str, object] = {
        "suite": suite,
        "query_id": q["query_id"],
        "n_vars": q["n_vars"],
        "catalogue": name,
    }
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TASK_TIMEOUT_S)
    try:
        query, data = SUITES[suite][1](q)
        attrs = sorted(query.all_attributes)
        blocks = enumerate_blocks(query, max_block_size=2)
        singles = [b for b in blocks if len(b.relations) == 1]
        pairs = [b for b in blocks if len(b.relations) == 2]
        row["rss_loaded_mb"] = f"{_rss_mb():.0f}"
        t0 = time.perf_counter()
        st = collect(name, query, data, singles, pairs)
        t1 = time.perf_counter()
        cons = block_constraints(
            query,
            data,
            max_block_size=1 if name == "lpbound" else 2,
            p_values=P11,
            lp_stats=st,
        )
        t2 = time.perf_counter()
        row["log2_bound"] = f"{polymatroid_bound(attrs, cons):.6f}"
        t3 = time.perf_counter()
        row["collect_s"] = f"{t1 - t0:.4f}"
        row["constraints_s"] = f"{t2 - t1:.4f}"
        row["solve_s"] = f"{t3 - t2:.4f}"
        if name == "lpbound":
            t4 = time.perf_counter()
            row["log2_flow"] = f"{flow_bound(attrs, cons).log_bound:.6f}"
            row["flow_s"] = f"{time.perf_counter() - t4:.4f}"
        row["rss_peak_mb"] = f"{_rss_mb():.0f}"
        row["status"] = "ok"
    except _TaskTimeoutError:
        row["status"] = f"over {TASK_TIMEOUT_S} s"
    finally:
        signal.alarm(0)
    return row


def _truth() -> dict[str, float]:
    out: dict[str, float] = {}
    for path in TRUTH.values():
        for r in csv.DictReader(path.read_text().splitlines()):
            if r["true_count"] not in ("", "None", "0"):
                out[r["query_id"]] = float(r["true_count"])
    return out


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    truth = _truth()
    by: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for r in rows:
        if r["status"] == "ok":
            by.setdefault((str(r["suite"]), str(r["query_id"])), {})[
                str(r["catalogue"])
            ] = r

    def f(r: dict[str, object], c: str) -> float:
        return float(str(r[c]))

    out: list[dict[str, object]] = []
    for suite in ("hetionet", "imdb"):
        qs = [v for (s, _), v in by.items() if s == suite and len(v) == len(CATALOGUES)]
        for name in CATALOGUES:
            chk = [v for v in qs if str(v[name]["query_id"]) in truth]
            lb = [f(v["lpbound"], "log2_bound") - f(v[name], "log2_bound") for v in qs]
            full = [
                f(v["lpbound"], "log2_bound") - f(v["full"], "log2_bound") for v in qs
            ]
            out.append(
                {
                    "suite": suite,
                    "catalogue": name,
                    "n": len(qs),
                    "median_collect_s": f"{statistics.median(f(v[name], 'collect_s') for v in qs):.4g}",
                    "median_collect_over_lpbound": f"{statistics.median(f(v[name], 'collect_s') / f(v['lpbound'], 'collect_s') for v in qs):.4g}",
                    "median_total_s": f"{statistics.median(f(v[name], 'collect_s') + f(v[name], 'constraints_s') + f(v[name], 'solve_s') for v in qs):.4g}",
                    "median_constraints_s": f"{statistics.median(f(v[name], 'constraints_s') for v in qs):.4g}",
                    "median_solve_s": f"{statistics.median(f(v[name], 'solve_s') for v in qs):.4g}",
                    "median_peak_over_loaded_mb": f"{statistics.median(f(v[name], 'rss_peak_mb') - f(v[name], 'rss_loaded_mb') for v in qs):.0f}",
                    "max_peak_mb": f"{max(f(v[name], 'rss_peak_mb') for v in qs):.0f}",
                    "median_gain_over_lpbound": f"{statistics.median(2**x for x in lb):.4g}",
                    "median_share_of_full_log_gain": f"{statistics.median(x / max(y, 1e-9) for x, y in zip(lb, full, strict=True)):.4g}",
                    "n_checked": len(chk),
                    "median_bound_over_truth": f"{statistics.median(2 ** f(v[name], 'log2_bound') / truth[str(v[name]['query_id'])] for v in chk):.4g}",
                    "n_invalid": sum(
                        2 ** f(v[name], "log2_bound")
                        < truth[str(v[name]["query_id"])] * (1 - 1e-9)
                        for v in chk
                    ),
                }
            )
        diff = [
            abs(f(v["sparse_full"], "log2_bound") - f(v["full"], "log2_bound"))
            for v in qs
        ]
        out.append(
            {
                "suite": suite,
                "catalogue": "checks",
                "n": len(qs),
                "max_abs_log2_sparse_full_minus_full": f"{max(diff):.2e}",
                "median_lpbound_flow_s": f"{statistics.median(f(v['lpbound'], 'flow_s') for v in qs):.4g}",
                "max_abs_log2_flow_minus_poly_bs1": f"{max(abs(f(v['lpbound'], 'log2_flow') - f(v['lpbound'], 'log2_bound')) for v in qs):.2e}",
                "n_tasks_not_ok": sum(
                    1 for r in rows if r["suite"] == suite and r["status"] != "ok"
                ),
            }
        )
    return out


def _write(name: str, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = [
        (suite, q, name)
        for suite in ("hetionet", "imdb")
        for q in json.loads(SUITES[suite][0].read_text())["queries"]
        for name in CATALOGUES
    ]
    random.Random(SEED).shuffle(items)
    with multiprocessing.Pool(processes=WORKERS, maxtasksperchild=1) as pool:
        rows = pool.map(evaluate, items, chunksize=1)
    rows.sort(key=lambda r: (str(r["suite"]), str(r["query_id"]), str(r["catalogue"])))
    _write("cost_accuracy_indep.csv", rows)
    summary = summarise(rows)
    _write("cost_accuracy_indep_summary.csv", summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
