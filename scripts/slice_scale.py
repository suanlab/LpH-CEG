"""Do the Hetionet long-suite cover-LP margins hold on a larger slice?

The long suite caps each relation at its first 20K rows in file order
(``evaluate_query_suite.MAX_ROWS``).  This reruns the |V|=6 queries with the
same file-order rule at 20K (the reference, through this same code path)
and at 30K rows, with eleven norms on both sides as in ``pset_fair.py``, and
reports the headline ratio of Table long-suite: the bs<=2 cover LP over the
entropic bound (LpBound).  The flow LP is left out: at 20K its eleven-norm run
already takes a median 403 s per query, and block-statistics collection
grows faster than the data (a 50K-row trial did not finish one query's
statistics in 20 min), which is the limit this experiment probes.

Each query runs under a wall-clock cap (``TASK_TIMEOUT_S``); a query over
the cap is recorded as such, not dropped.  Each query is also counted
exactly by DuckDB (``EXACT_TIMEOUT_S``) so every bound with a count is
checked.

Outputs: results/suite/slice_scale.csv and slice_scale_summary.csv
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import signal
import statistics
import sys
import time
from pathlib import Path

import scripts.evaluate_query_suite as eqs
from scripts import pset_fair
from scripts.evaluate_long_suite import exact_count
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import block_constraints, flow_bound
from src.estimators.lph_ceg import estimate_lph_ceg
from src.statistics.lp_stats import compute_all_lp_block_extensions

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "suite"
CAPS: tuple[int, ...] = (20_000, 30_000)
N_VARS = 6
WORKERS = 8
TASK_TIMEOUT_S = 1800
EXACT_TIMEOUT_S = 120.0
P = pset_fair.P11


class _TaskTimeoutError(Exception):
    pass


def _alarm(signum, frame):  # type: ignore[no-untyped-def]
    raise _TaskTimeoutError


def evaluate(item: tuple[int, dict]) -> dict[str, object]:  # type: ignore[type-arg]
    cap, q = item
    eqs.MAX_ROWS = cap
    row: dict[str, object] = {"cap": cap, "query_id": q["query_id"]}
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TASK_TIMEOUT_S)
    t0 = time.perf_counter()
    try:
        built = eqs.build_hetionet_query(q)
        assert built is not None
        query, data = built
        row["max_relation_rows"] = max(len(rows) for _, rows in data.values())
        attrs = sorted(query.all_attributes)
        blocks = enumerate_blocks(query, max_block_size=2)
        stats = compute_all_lp_block_extensions(query, blocks, data, p_values=P)
        cons1 = block_constraints(
            query, data, max_block_size=1, p_values=P, lp_stats=stats
        )
        row["entropic_p11"] = f"{flow_bound(attrs, cons1).bound:.6e}"
        row["cover_bs2_p11"] = (
            f"{estimate_lph_ceg(query, data, max_block_size=2, p_values=P, lp_stats=stats).estimate:.6e}"
        )
        row["seconds"] = f"{time.perf_counter() - t0:.1f}"
        signal.alarm(0)
        truth = exact_count(query, data, EXACT_TIMEOUT_S)
        row["true_count"] = "" if truth is None else truth
        row["status"] = "ok"
    except _TaskTimeoutError:
        row["status"] = f"over {TASK_TIMEOUT_S} s"
        row["seconds"] = f"{time.perf_counter() - t0:.1f}"
    finally:
        signal.alarm(0)
    return row


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def f(r: dict[str, object], c: str) -> float:
        return float(str(r[c]))

    out: list[dict[str, object]] = []
    for cap in sorted({int(str(r["cap"])) for r in rows}):
        rs = [r for r in rows if int(str(r["cap"])) == cap]
        ok = [r for r in rs if r["status"] == "ok"]
        chk = [r for r in ok if str(r.get("true_count", "")) not in ("", "0")]
        out.append(
            {
                "cap": cap,
                "n": len(rs),
                "n_ok": len(ok),
                "median_entropic_over_cover_bs2_p11": f"{statistics.median(f(r, 'entropic_p11') / f(r, 'cover_bs2_p11') for r in ok):.4g}",
                "n_cover_bs2_below_entropic_p11": sum(
                    f(r, "cover_bs2_p11") < f(r, "entropic_p11") for r in ok
                ),
                "n_checked": len(chk),
                "n_invalid": sum(
                    f(r, b) < f(r, "true_count") * (1 - 1e-9)
                    for r in chk
                    for b in ("entropic_p11", "cover_bs2_p11")
                ),
                "median_seconds": f"{statistics.median(f(r, 'seconds') for r in ok):.1f}",
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
    caps = tuple(int(a) for a in sys.argv[1:]) or CAPS
    qpath = pset_fair.SUITES["hetionet"][0]
    queries = [
        q for q in json.loads(qpath.read_text())["queries"] if q["n_vars"] == N_VARS
    ]
    items = [(cap, q) for cap in caps for q in queries]
    with multiprocessing.Pool(processes=WORKERS, maxtasksperchild=1) as pool:
        rows = pool.map(evaluate, items, chunksize=1)
    rows.sort(key=lambda r: (int(str(r["cap"])), str(r["query_id"])))
    _write("slice_scale.csv", rows)
    summary = summarise(rows)
    _write("slice_scale_summary.csv", summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
