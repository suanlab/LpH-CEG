"""Is the tail of robustness_broad.py stable under repeated, shuffled timing?

``robustness_broad.py`` times each sampled forced order once, the
bound-guided order and DuckDB's plan three times, and divides by the
fastest forced time it saw.  This repeats the same 24 queries and the same
plan sample (same seeds) with every plan timed the same number of times,
in a freshly shuffled order in every repetition, after one warm-up run of
each plan.  The reference of a query is the smallest *median* time over
its forced orders, so selecting the reference and evaluating against it do
not share a single noisy run.  Per repetition it also reports the P95 and
maximum across queries (P95 as in robustness_broad.py: index
floor(0.95 (n-1)) of the sorted values), to show how much the tail moves.

Run alone: timings are wall-clock and other load distorts them.
Output: results/e2e/robustness_repeat.csv and _summary.csv
"""

from __future__ import annotations

import csv
import random
import statistics
import time
from pathlib import Path

import scripts.robustness_broad as rb
from scripts.e2e_plan_injection import (
    build_sql,
    choose_order,
    connected_orders,
    load_duckdb,
)
from src.estimators.agm_bound import estimate_agm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "e2e"
REPS = 5


def _time(con, sql: str, force: bool) -> float:  # type: ignore[no-untyped-def]
    con.execute(
        "SET disabled_optimizers='join_order'"
        if force
        else "SET disabled_optimizers=''"
    )
    t0 = time.perf_counter()
    con.execute(sql).fetchone()
    return (time.perf_counter() - t0) * 1000


def main() -> None:
    suite = [
        (shape, edges, nv, ne, seed)
        for shape, edges in rb.SHAPES.items()
        for nv, ne in ((80, 300), (110, 480))
        for seed in (1, 2, 3)
    ]
    rng = random.Random(0)  # same plan sample as robustness_broad.py
    shuffle = random.Random(2026)
    rows: list[dict[str, object]] = []
    per_rep: dict[int, list[tuple[float, float]]] = {r: [] for r in range(REPS)}
    for shape, edges, nv, ne, seed in suite:
        q, data = rb.build_seeded(edges, nv, ne, seed)
        if estimate_agm(q, data) > rb.CAP:
            continue
        orders = connected_orders(q)
        bound_order = choose_order(q, data, orders, worst=False)
        sample = orders if len(orders) <= 16 else rng.sample(orders, 16)
        eval_orders = sorted({bound_order, *sample})
        con = load_duckdb(q, data)
        plans = [("forced", o) for o in eval_orders] + [("duckdb", bound_order)]
        counts = set()
        for kind, o in plans:  # warm-up, and every plan must count the same join
            if kind == "duckdb":
                con.execute("SET disabled_optimizers=''")
            else:
                con.execute("SET disabled_optimizers='join_order'")
            counts.add(con.execute(build_sql(q, o)).fetchone()[0])
        if len(counts) != 1:
            con.close()
            continue
        times: dict[tuple[str, tuple[str, ...]], list[float]] = {p: [] for p in plans}
        for _ in range(REPS):
            order = plans[:]
            shuffle.shuffle(order)
            for kind, o in order:
                times[(kind, o)].append(_time(con, build_sql(q, o), kind == "forced"))
        con.close()
        med = {p: statistics.median(ts) for p, ts in times.items()}
        ref = min(v for (k, _), v in med.items() if k == "forced")
        bound_slow = med[("forced", bound_order)] / ref
        duck_slow = med[("duckdb", bound_order)] / ref
        for r in range(REPS):
            ref_r = min(times[("forced", o)][r] for o in eval_orders)
            per_rep[r].append(
                (
                    times[("forced", bound_order)][r] / ref_r,
                    times[("duckdb", bound_order)][r] / ref_r,
                )
            )
        rows.append(
            {
                "shape": shape,
                "nv": nv,
                "ne": ne,
                "seed": seed,
                "n_orders": len(eval_orders),
                "ref_ms": f"{ref:.3f}",
                "bound_slowdown": f"{bound_slow:.4f}",
                "duckdb_slowdown": f"{duck_slow:.4f}",
            }
        )
        print(rows[-1], flush=True)
    with open(OUT / "robustness_repeat.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bg = [float(str(r["bound_slowdown"])) for r in rows]
    dk = [float(str(r["duckdb_slowdown"])) for r in rows]
    summary = [
        {
            "scope": "median-of-5 timings",
            "n": len(rows),
            "bound_median": f"{statistics.median(bg):.4f}",
            "bound_p95": f"{rb.pctl(bg, 0.95):.4f}",
            "bound_max": f"{max(bg):.4f}",
            "duckdb_median": f"{statistics.median(dk):.4f}",
            "duckdb_p95": f"{rb.pctl(dk, 0.95):.4f}",
            "duckdb_max": f"{max(dk):.4f}",
        }
    ]
    for r, pairs in per_rep.items():
        b = [x for x, _ in pairs]
        d = [y for _, y in pairs]
        summary.append(
            {
                "scope": f"rep {r + 1}",
                "n": len(pairs),
                "bound_median": f"{statistics.median(b):.4f}",
                "bound_p95": f"{rb.pctl(b, 0.95):.4f}",
                "bound_max": f"{max(b):.4f}",
                "duckdb_median": f"{statistics.median(d):.4f}",
                "duckdb_p95": f"{rb.pctl(d, 0.95):.4f}",
                "duckdb_max": f"{max(d):.4f}",
            }
        )
    with open(OUT / "robustness_repeat_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
