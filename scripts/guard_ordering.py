"""A plan guard from the bound: does it catch the engine's catastrophic plans?

Bound-guided left-deep orders avoid the engine's catastrophic plans on dense
cliques but lose to DuckDB's own (bushy) plans on average and on IMDb.  A
guard keeps DuckDB's plan unless the pessimistic bound on the whole query
exceeds DuckDB's own estimate of the final join by more than a factor tau,
and then forces the bound-guided order.

No query is re-executed.  The slowdowns of both plans were measured by
``robustness_broad.py`` (24 synthetic cyclic queries, bs=1 cover LP) and
``imdb_plan_injection.py`` (IMDb |V|=6, bs<=2 flow LP); here we only add
the guard's signal: DuckDB's EXPLAIN estimate of the root join (the same
DuckDB version, optimizer on; the first ``~N rows`` of the plan text) and
the bound the strategy used.  Every tau of a grid fixed in advance is
reported -- none is selected.

Result (negative): the guard does not trim the tail.  On the synthetic
queries where DuckDB's plan is catastrophic its root estimate is far
*above* the true count, so a root-level underestimate signal cannot see
them; the failure lies in intermediate estimates (EXPLAIN ANALYZE on two
K4 instances: an intermediate join estimated at 6,258 rows produces
241,560, and 2,337 produces 54,242).  Not used in the paper.

Output: results/e2e/guard_ordering.csv and guard_ordering_summary.csv
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import scripts.robustness_broad as rb
from scripts.e2e_plan_injection import (
    build_sql,
    connected_orders,
    estimate_cover,
    load_duckdb,
)
from scripts.evaluate_imdb_suite import build_imdb_query
from scripts.pset_fair import SUITES

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "e2e"
TAUS: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)


def duckdb_root_estimate(query, data) -> float:  # type: ignore[no-untyped-def]
    """DuckDB's estimated cardinality of the top join (EXPLAIN, optimizer on)."""
    con = load_duckdb(query, data)
    con.execute("SET disabled_optimizers=''")
    plan = con.execute("EXPLAIN " + build_sql(query, connected_orders(query)[0]))
    text = plan.fetchall()[0][1]
    con.close()
    m = re.search(r"~([\d,]+) rows", text)
    if m is None:
        raise ValueError("no cardinality estimate in the plan")
    return float(m.group(1).replace(",", ""))


def synthetic_rows() -> list[dict[str, object]]:
    rows = []
    for r in csv.DictReader((OUT / "robustness_broad.csv").read_text().splitlines()):
        q, data = rb.build_seeded(
            rb.SHAPES[r["shape"]], int(r["nv"]), int(r["ne"]), int(r["seed"])
        )
        names = tuple(rel.name for rel in q.relations)
        rows.append(
            {
                "suite": "synthetic",
                "query": f"{r['shape']} {r['nv']}V/{r['ne']}E s{r['seed']}",
                "true_count": r["count"],
                "bound": estimate_cover(q, data, names),
                "duckdb_estimate": duckdb_root_estimate(q, data),
                "bound_slowdown": float(r["bound_slowdown"]),
                "duckdb_slowdown": float(r["duckdb_slowdown"]),
            }
        )
    return rows


def imdb_rows() -> list[dict[str, object]]:
    reg = {
        q["query_id"]: q for q in json.loads(SUITES["imdb"][0].read_text())["queries"]
    }
    flow = {
        r["query_id"]: float(r["flow_bs2"])
        for r in csv.DictReader(
            (ROOT / "results" / "imdb" / "imdb_suite_eval.csv").read_text().splitlines()
        )
    }
    rows = []
    for r in csv.DictReader(
        (ROOT / "results" / "imdb" / "imdb_plan_injection.csv").read_text().splitlines()
    ):
        if r["n_timeouts"] != "0":
            continue  # the 33 whose forced orders all finish, as in the paper
        q, data = build_imdb_query(reg[r["query_id"]])
        rows.append(
            {
                "suite": "imdb",
                "query": r["query_id"],
                "true_count": r["count"],
                "bound": flow[r["query_id"]],
                "duckdb_estimate": duckdb_root_estimate(q, data),
                "bound_slowdown": float(r["flow_bs2_slowdown"]),
                "duckdb_slowdown": float(r["duckdb_slowdown"]),
            }
        )
    return rows


def pctl(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def main() -> None:
    rows = synthetic_rows() + imdb_rows()
    for r in rows:
        r["signal"] = float(str(r["bound"])) / max(
            float(str(r["duckdb_estimate"])), 1.0
        )
    with open(OUT / "guard_ordering.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary: list[dict[str, object]] = []
    for suite in ("synthetic", "imdb"):
        rs = [r for r in rows if r["suite"] == suite]
        plans = {
            "duckdb": [float(str(r["duckdb_slowdown"])) for r in rs],
            "bound": [float(str(r["bound_slowdown"])) for r in rs],
        }
        fired = {}
        for tau in TAUS:
            pick = [float(str(r["signal"])) > tau for r in rs]
            fired[tau] = sum(pick)
            plans[f"guard_tau{tau:g}"] = [
                float(str(r["bound_slowdown"] if p else r["duckdb_slowdown"]))
                for r, p in zip(rs, pick, strict=True)
            ]
        for name, xs in plans.items():
            tau = float(name.split("tau")[1]) if "tau" in name else None
            summary.append(
                {
                    "suite": suite,
                    "plan": name,
                    "n": len(xs),
                    "n_guard_fired": "" if tau is None else fired[tau],
                    "median": f"{pctl(xs, 0.5):.3f}",
                    "p95": f"{pctl(xs, 0.95):.3f}",
                    "max": f"{max(xs):.3f}",
                }
            )
    with open(OUT / "guard_ordering_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
