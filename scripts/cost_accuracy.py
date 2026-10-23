"""What does each statistic cost, and what does it buy, in one setting?

Same queries (Hetionet and IMDb long suites, |V| = 6, 8, 10), same norms
(eleven, P = {1,...,10,inf}) and the same inference (the 2^|V| polymatroid
program) for four statistics catalogues:

  lpbound    single-relation statistics (LpBound's input)
  free       + the join-free interfaces of every pair block (pushdown)
  free_l2    + the exact l2 norm of each block's common-neighbour counts,
             computed as ||A_R^T A_S||_F by one sparse product (no other
             statistic of the pair join)
  full       + every interface of every pair block (the paper's bs<=2)

For each catalogue: cumulative statistics-collection seconds (cold, per
query, nothing shared across queries), polymatroid solve seconds, peak RSS
after the stage, the bound's gain over LpBound and its ratio to the exact
count where one exists.  LpBound's production inference, its flow LP, is
also timed.  All stages of a query run in one process, so per-query ratios
are robust to machine load; absolute times are from our Python prototype
with eight worker processes.

Sanity check: free_l2 must reproduce ablation_p11_poly.csv's free_l2 bound.
Output: results/cost/cost_accuracy.csv and cost_accuracy_summary.csv
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing
import resource
import signal
import statistics
import time
from pathlib import Path

from scipy.sparse import csr_matrix

from scripts.pset_fair import P11, SUITES
from scripts.pushdown_stats_experiment import _l2_only
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import block_constraints, flow_bound, polymatroid_bound
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.statistics.pushdown_stats import (
    _binary_block,
    codegree_interface,
    compute_pushdown_block_extensions,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cost"
WORKERS = 8
TASK_TIMEOUT_S = 3600
CATALOGUES = ("lpbound", "free", "free_l2", "full")
TRUTH = {
    "hetionet": ROOT / "results" / "suite" / "long_suite_eval.csv",
    "imdb": ROOT / "results" / "imdb" / "imdb_suite_eval.csv",
}


class _TaskTimeoutError(Exception):
    pass


def _alarm(signum, frame):  # type: ignore[no-untyped-def]
    raise _TaskTimeoutError


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def codegree_l2_exact(query, block, data) -> float:  # type: ignore[no-untyped-def]
    """||A_R^T A_S||_F: the l2 norm of the common-neighbour counts."""
    rows_r, (ia, ib_r), rows_s, (ic, ib_s) = _binary_block(query, block, data)
    a_ids: dict[str, int] = {}
    b_ids: dict[str, int] = {}
    c_ids: dict[str, int] = {}
    ra = [a_ids.setdefault(r[ia], len(a_ids)) for r in rows_r]
    rb = [b_ids.setdefault(r[ib_r], len(b_ids)) for r in rows_r]
    sb = [b_ids.setdefault(r[ib_s], len(b_ids)) for r in rows_s]
    sc = [c_ids.setdefault(r[ic], len(c_ids)) for r in rows_s]
    nb = len(b_ids)
    mr = csr_matrix(([1.0] * len(ra), (ra, rb)), shape=(len(a_ids), nb))
    ms = csr_matrix(([1.0] * len(sb), (sb, sc)), shape=(nb, len(c_ids)))
    prod = mr @ ms
    return math.sqrt(float((prod.data**2).sum()))


def evaluate(item: tuple[str, dict]) -> dict[str, object]:  # type: ignore[type-arg]
    suite, q = item
    row: dict[str, object] = {
        "suite": suite,
        "query_id": q["query_id"],
        "n_vars": q["n_vars"],
    }
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TASK_TIMEOUT_S)
    try:
        query, data = SUITES[suite][1](q)
        attrs = sorted(query.all_attributes)
        blocks = enumerate_blocks(query, max_block_size=2)
        singles = [b for b in blocks if len(b.relations) == 1]
        pairs = [b for b in blocks if len(b.relations) == 2]

        t0 = time.perf_counter()
        st1 = compute_all_lp_block_extensions(query, singles, data, p_values=P11)
        t_single = time.perf_counter() - t0
        row["rss_lpbound_mb"] = f"{_rss_mb():.0f}"

        t0 = time.perf_counter()
        free = dict(st1)
        for b in pairs:
            free.update(compute_pushdown_block_extensions(query, b, data, p_values=P11))
        t_free = time.perf_counter() - t0
        row["rss_free_mb"] = f"{_rss_mb():.0f}"

        t0 = time.perf_counter()
        free_l2 = dict(free)
        for b in pairs:
            free_l2[(b, codegree_interface(query, b))] = _l2_only(
                codegree_l2_exact(query, b, data)
            )
        t_l2 = time.perf_counter() - t0
        row["rss_free_l2_mb"] = f"{_rss_mb():.0f}"

        t0 = time.perf_counter()
        full = dict(st1)
        full.update(compute_all_lp_block_extensions(query, pairs, data, p_values=P11))
        t_full = time.perf_counter() - t0
        row["rss_full_mb"] = f"{_rss_mb():.0f}"

        collect = {
            "lpbound": t_single,
            "free": t_single + t_free,
            "free_l2": t_single + t_free + t_l2,
            "full": t_single + t_full,
        }
        cats = {"lpbound": st1, "free": free, "free_l2": free_l2, "full": full}
        for name, st in cats.items():
            cons = block_constraints(
                query,
                data,
                max_block_size=1 if name == "lpbound" else 2,
                p_values=P11,
                lp_stats=st,
            )
            t0 = time.perf_counter()
            row[f"log2_{name}"] = f"{polymatroid_bound(attrs, cons):.6f}"
            row[f"infer_{name}_s"] = f"{time.perf_counter() - t0:.4f}"
            row[f"collect_{name}_s"] = f"{collect[name]:.4f}"
            if name == "lpbound":
                t0 = time.perf_counter()
                row["log2_lpbound_flow"] = f"{flow_bound(attrs, cons).log_bound:.6f}"
                row["infer_lpbound_flow_s"] = f"{time.perf_counter() - t0:.4f}"
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

    def f(r: dict[str, object], c: str) -> float:
        return float(str(r[c]))

    out: list[dict[str, object]] = []
    for suite in ("hetionet", "imdb"):
        rs = [r for r in rows if r["suite"] == suite and r["status"] == "ok"]
        for name in CATALOGUES:
            chk = [r for r in rs if str(r["query_id"]) in truth]
            out.append(
                {
                    "suite": suite,
                    "catalogue": name,
                    "n": len(rs),
                    "median_collect_s": f"{statistics.median(f(r, f'collect_{name}_s') for r in rs):.4g}",
                    "median_collect_over_lpbound": f"{statistics.median(f(r, f'collect_{name}_s') / f(r, 'collect_lpbound_s') for r in rs):.4g}",
                    "median_infer_s": f"{statistics.median(f(r, f'infer_{name}_s') for r in rs):.4g}",
                    "median_peak_rss_mb": f"{statistics.median(f(r, f'rss_{name}_mb') for r in rs):.0f}",
                    "median_gain_over_lpbound": f"{statistics.median(2 ** (f(r, 'log2_lpbound') - f(r, f'log2_{name}')) for r in rs):.4g}",
                    "median_share_of_full_log_gain": f"{statistics.median((f(r, 'log2_lpbound') - f(r, f'log2_{name}')) / max(f(r, 'log2_lpbound') - f(r, 'log2_full'), 1e-9) for r in rs):.4g}",
                    "n_checked": len(chk),
                    "median_bound_over_truth": f"{statistics.median(2 ** f(r, f'log2_{name}') / truth[str(r['query_id'])] for r in chk):.4g}",
                    "n_invalid": sum(
                        2 ** f(r, f"log2_{name}")
                        < truth[str(r["query_id"])] * (1 - 1e-9)
                        for r in chk
                    ),
                }
            )
        out.append(
            {
                "suite": suite,
                "catalogue": "lpbound_flow",
                "n": len(rs),
                "median_infer_s": f"{statistics.median(f(r, 'infer_lpbound_flow_s') for r in rs):.4g}",
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
        (suite, q)
        for suite in ("hetionet", "imdb")
        for q in json.loads(SUITES[suite][0].read_text())["queries"]
    ]
    with multiprocessing.Pool(processes=WORKERS, maxtasksperchild=1) as pool:
        rows = pool.map(evaluate, items, chunksize=1)
    rows.sort(key=lambda r: (str(r["suite"]), str(r["query_id"])))
    ref = {
        (r["suite"], r["query_id"]): float(r["log2_free_l2"])
        for r in csv.DictReader(
            (ROOT / "results" / "pushdown" / "ablation_p11_poly.csv")
            .read_text()
            .splitlines()
        )
        if r["status"] == "ok"
    }
    for r in rows:
        key = (r["suite"], r["query_id"])
        if r["status"] == "ok" and key in ref:
            r["free_l2_vs_ablation"] = f"{float(str(r['log2_free_l2'])) - ref[key]:.2e}"
    _write("cost_accuracy.csv", rows)
    summary = summarise(rows)
    _write("cost_accuracy_summary.csv", summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
