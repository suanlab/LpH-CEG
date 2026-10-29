"""Replication of robustness_repeat.py that keeps every timing.

Same 24 queries and plan sample (same seeds) as ``robustness_repeat.py``,
which stores only per-query summaries.  Here every observation
``(query, plan, repetition, position in the shuffled run, ms)`` is written
out, and the reference plan is chosen and evaluated on disjoint
repetitions: repetitions 1--2 pick the fastest forced order (smallest
median), repetitions 3--6 evaluate it, so choosing the reference and
measuring against it do not share a run.  The summary reports the pooled
tail, the tail with the split reference, and per-shape medians of the
paired bound-guided / DuckDB time ratios.

Run alone: timings are wall-clock and other load distorts them.
Output: results/e2e/robustness_raw.csv and robustness_raw_summary.csv
"""

from __future__ import annotations

import csv
import random
import statistics
from pathlib import Path

import scripts.robustness_broad as rb
from scripts.e2e_plan_injection import (
    build_sql,
    choose_order,
    connected_orders,
    load_duckdb,
)
from scripts.robustness_repeat import _time
from src.estimators.agm_bound import estimate_agm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "e2e"
REPS = 6
SELECT = (0, 1)  # repetitions used to choose the reference plan


def main() -> None:
    suite = [
        (shape, edges, nv, ne, seed)
        for shape, edges in rb.SHAPES.items()
        for nv, ne in ((80, 300), (110, 480))
        for seed in (1, 2, 3)
    ]
    rng = random.Random(0)  # same plan sample as robustness_broad.py
    shuffle = random.Random(4049)
    obs: list[dict[str, object]] = []
    per_query: list[dict[str, object]] = []
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
        for kind, o in plans:  # warm-up; every plan must count the same join
            con.execute(
                "SET disabled_optimizers=''"
                if kind == "duckdb"
                else "SET disabled_optimizers='join_order'"
            )
            counts.add(con.execute(build_sql(q, o)).fetchone()[0])
        if len(counts) != 1:
            con.close()
            continue
        qid = f"{shape}-{nv}-{seed}"
        times: dict[tuple[str, tuple[str, ...]], list[float]] = {p: [] for p in plans}
        for r in range(REPS):
            order = plans[:]
            shuffle.shuffle(order)
            for pos, (kind, o) in enumerate(order):
                ms = _time(con, build_sql(q, o), kind == "forced")
                times[(kind, o)].append(ms)
                obs.append(
                    {
                        "query": qid,
                        "shape": shape,
                        "plan": f"{kind}:{'-'.join(o)}",
                        "is_bound_order": o == bound_order,
                        "rep": r + 1,
                        "position": pos,
                        "ms": f"{ms:.4f}",
                    }
                )
        con.close()

        def med(
            p: tuple[str, tuple[str, ...]],
            reps: tuple[int, ...],
            times: dict[tuple[str, tuple[str, ...]], list[float]] = times,
        ) -> float:
            return statistics.median(times[p][r] for r in reps)

        all_reps = tuple(range(REPS))
        eval_reps = tuple(r for r in all_reps if r not in SELECT)
        forced = [p for p in plans if p[0] == "forced"]
        ref_all = min(med(p, all_reps) for p in forced)
        ref_plan = min(forced, key=lambda p: med(p, SELECT))
        ref_split = med(ref_plan, eval_reps)
        b, d = ("forced", bound_order), ("duckdb", bound_order)
        per_query.append(
            {
                "query": qid,
                "shape": shape,
                "bound_slowdown": med(b, all_reps) / ref_all,
                "duckdb_slowdown": med(d, all_reps) / ref_all,
                "bound_slowdown_split": med(b, eval_reps) / ref_split,
                "duckdb_slowdown_split": med(d, eval_reps) / ref_split,
                "bound_over_duckdb": med(b, all_reps) / med(d, all_reps),
            }
        )
        print(per_query[-1], flush=True)
    with open(OUT / "robustness_raw.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(obs[0]))
        w.writeheader()
        w.writerows(obs)

    def stats(rows: list[dict[str, object]], col: str) -> dict[str, str]:
        xs = [float(str(r[col])) for r in rows]
        return {
            f"{col}_median": f"{statistics.median(xs):.4f}",
            f"{col}_p95": f"{rb.pctl(xs, 0.95):.4f}",
            f"{col}_max": f"{max(xs):.4f}",
        }

    summary: list[dict[str, object]] = []
    for scope in ("all", *sorted({str(r["shape"]) for r in per_query})):
        rows = [r for r in per_query if scope in ("all", r["shape"])]
        rec: dict[str, object] = {"scope": scope, "n": len(rows)}
        for col in (
            "bound_slowdown",
            "duckdb_slowdown",
            "bound_slowdown_split",
            "duckdb_slowdown_split",
            "bound_over_duckdb",
        ):
            rec.update(stats(rows, col))
        summary.append(rec)
    with open(OUT / "robustness_raw_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
