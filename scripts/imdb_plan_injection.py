"""Bound-guided join ordering on real data with block statistics (IMDb, DuckDB).

The robustness experiment of `e2e_plan_injection.py` used synthetic queries
and the bs=1 cover bound only, so it says nothing about the paper's thesis
on real data.  This runs the same protocol on the 40 registered IMDb cycles
at |V| = 6 (`register_imdb_suite.py`, 200 most-voted films):

1. enumerate the connected left-deep join orders of the query;
2. for each strategy pick the order minimising C_out, the sum of the
   strategy's estimates over the order's prefixes (sub-paths of the
   cycle), and force it in DuckDB (``disabled_optimizers='join_order'``);
3. time a deterministic sample of other orders to estimate the best and
   worst achievable runtimes, and run DuckDB's own optimizer.

Strategies: ``cover_bs1`` (our bs=1 cover LP), ``lpbound`` (the entropic
bound on bs=1 statistics, via the flow LP), ``flow_bs2`` (the flow LP over
the bs<=2 block catalogue -- the paper's recommended estimator) and
``oracle`` (exact prefix counts), plus ``duckdb`` (optimizer on).  Each
execution has a TIMEOUT_S budget; a timed-out run is recorded at the budget
and flagged (a lower bound on its slowdown).  Every order must return the
same count.

Outputs: results/imdb/imdb_plan_injection.csv and _summary.csv.

Usage:
    PYTHONPATH=. python scripts/imdb_plan_injection.py
"""

from __future__ import annotations

import csv
import multiprocessing
import random
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.e2e_plan_injection import build_sql, connected_orders, load_duckdb
from scripts.evaluate_imdb_suite import P_VALUES, build_imdb_query, verify_registration
from src.estimators.flow_bound import (
    block_constraints,
    cycle_candidate_orders,
    flow_bound,
    flow_bound_over_orders,
)
from src.estimators.lph_ceg import estimate_lph_ceg
from src.utils.types import QueryHypergraph

if TYPE_CHECKING:
    import duckdb

    from src.statistics.block_stats import RelationDataMap

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "imdb"
TIMEOUT_S = 60.0
SAMPLE = 16  # extra orders timed to estimate the achievable range
WORKERS = 10
DUCKDB_THREADS = 4
STRATEGIES = ("cover_bs1", "lpbound", "flow_bs2", "oracle")


def _sub(query: QueryHypergraph, names: tuple[str, ...]) -> QueryHypergraph:
    rels = tuple(r for r in query.relations if r.name in names)
    return QueryHypergraph(relations=rels, name="+".join(names))


def _path_order(sub: QueryHypergraph) -> tuple[str, ...]:
    """Attributes of a sub-path (or the cycle) in walk order."""
    attrs = sorted(sub.all_attributes, key=lambda a: int(a[1:]))
    # rotate so that a gap in the index sequence (path ends) comes last
    for i in range(len(attrs)):
        nxt = attrs[(i + 1) % len(attrs)]
        if int(nxt[1:]) != int(attrs[i][1:]) + 1:
            return tuple(attrs[i + 1 :] + attrs[: i + 1])
    return tuple(attrs)


def estimates(
    query: QueryHypergraph,
    data: RelationDataMap,
    names: tuple[str, ...],
    con: duckdb.DuckDBPyConnection,
) -> dict[str, float]:
    """All strategies' estimates for the sub-query on ``names`` (a connected order)."""
    sub = _sub(query, names)
    attrs = sorted(sub.all_attributes)
    cons1 = block_constraints(sub, data, max_block_size=1, p_values=P_VALUES)
    cons2 = block_constraints(sub, data, max_block_size=2, p_values=P_VALUES)
    orders = cycle_candidate_orders(_path_order(sub))
    con.execute("SET disabled_optimizers=''")
    return {
        "cover_bs1": estimate_lph_ceg(
            sub, data, max_block_size=1, p_values=P_VALUES
        ).estimate,
        "lpbound": flow_bound(attrs, cons1).bound,
        "flow_bs2": flow_bound_over_orders(attrs, cons2, orders)[0].bound,
        "oracle": float(con.execute(build_sql(sub, names)).fetchone()[0]),  # type: ignore[index]
    }


def run_timed(
    con: duckdb.DuckDBPyConnection, sql: str, force: bool
) -> tuple[int | None, float, bool]:
    """(count, ms, timed_out); a timed-out run is recorded at the budget."""
    con.execute(
        "SET disabled_optimizers='join_order'"
        if force
        else "SET disabled_optimizers=''"
    )
    out: list[int | None] = [None]

    def go() -> None:
        try:
            out[0] = int(con.execute(sql).fetchone()[0])  # type: ignore[index]
        except Exception:
            out[0] = None

    t0 = time.perf_counter()
    th = threading.Thread(target=go, daemon=True)
    th.start()
    th.join(TIMEOUT_S)
    if th.is_alive():
        con.interrupt()
        th.join()
        return None, TIMEOUT_S * 1000, True
    return out[0], (time.perf_counter() - t0) * 1000, False


def case(q: dict) -> dict[str, object]:  # type: ignore[type-arg]
    query, data = build_imdb_query(q)
    con = load_duckdb(query, data)
    con.execute(f"SET threads TO {DUCKDB_THREADS}")
    orders = connected_orders(query)
    cache: dict[tuple[str, ...], dict[str, float]] = {}
    t0 = time.perf_counter()

    def cout(o: tuple[str, ...], s: str) -> float:
        total = 0.0
        for k in range(2, len(o) + 1):
            key = tuple(sorted(o[:k]))
            if key not in cache:
                cache[key] = estimates(query, data, o[:k], con)
            total += cache[key][s]
        return total

    chosen = {s: min(orders, key=lambda o, s=s: cout(o, s)) for s in STRATEGIES}
    est_s = time.perf_counter() - t0
    rng = random.Random(q["query_id"])
    sample = rng.sample(orders, min(SAMPLE, len(orders)))
    timed: dict[tuple[str, ...], tuple[int | None, float, bool]] = {}
    for o in dict.fromkeys([*chosen.values(), *sample]):
        timed[o] = run_timed(con, build_sql(query, o), force=True)
    duck = run_timed(con, build_sql(query, chosen["flow_bs2"]), force=False)
    con.close()
    counts = {c for c, _, to in [*timed.values(), duck] if not to}
    assert len(counts) <= 1, f"{q['query_id']}: orders disagree {counts}"
    best = min(ms for _, ms, _ in timed.values())
    worst = max(ms for _, ms, _ in timed.values())
    row: dict[str, object] = {
        "query_id": q["query_id"],
        "n_orders": len(orders),
        "n_timed": len(timed),
        "count": next(iter(counts)) if counts else "",
        "best_ms": f"{best:.2f}",
        "worst_ms": f"{worst:.2f}",
        "spread": f"{worst / best:.4f}",
        "n_timeouts": sum(to for _, _, to in timed.values()),
        "estimate_s": f"{est_s:.1f}",
    }
    for s, o in chosen.items():
        _, ms, to = timed[o]
        row[f"{s}_slowdown"] = f"{ms / best:.4f}"
        row[f"{s}_timeout"] = int(to)
    row["duckdb_slowdown"] = f"{duck[1] / best:.4f}"
    row["duckdb_timeout"] = int(duck[2])
    row["flow_eq_cover_order"] = int(chosen["flow_bs2"] == chosen["cover_bs1"])
    row["flow_eq_oracle_order"] = int(chosen["flow_bs2"] == chosen["oracle"])
    return row


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def q95(v: list[float]) -> float:
        v = sorted(v)
        return v[min(len(v) - 1, int(0.95 * len(v)))]

    # Slowdowns are only meaningful when every forced order finished; a
    # query whose forced orders hit the budget is counted separately.
    done = [r for r in rows if int(str(r["n_timeouts"])) == 0]
    all_forced_out = sum(
        1 for r in rows if int(str(r["n_timeouts"])) >= int(str(r["n_timed"])) - 4
    )
    out: list[dict[str, object]] = []
    for s in (*STRATEGIES, "duckdb"):
        v = [float(str(r[f"{s}_slowdown"])) for r in done]
        out.append(
            {
                "strategy": s,
                "n": len(v),
                "median_slowdown": f"{statistics.median(v):.4f}",
                "p95_slowdown": f"{q95(v):.4f}",
                "max_slowdown": f"{max(v):.4f}",
                "n_within_1_5x": sum(x <= 1.5 for x in v),
                "n_over_2x": sum(x > 2 for x in v),
                "n_timeouts": sum(int(str(r[f"{s}_timeout"])) for r in rows),
                "n_queries": len(rows),
                "n_complete": len(done),
                "n_forced_mostly_timed_out": all_forced_out,
                "n_eq_oracle_order": sum(
                    int(str(r["flow_eq_oracle_order"])) for r in rows
                )
                if s == "flow_bs2"
                else "",
            }
        )
    return out


def _write(name: str, rows: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    if "--summary-only" in sys.argv:
        with open(OUT / "imdb_plan_injection.csv") as fh:
            _write(
                "imdb_plan_injection_summary.csv", summarise(list(csv.DictReader(fh)))
            )
        return
    reg = verify_registration()
    qs = [q for q in reg["queries"] if q["n_vars"] == 6]
    rows: list[dict[str, object]] = []
    with multiprocessing.Pool(processes=WORKERS, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(case, qs):
            rows.append(r)
            rows.sort(key=lambda x: str(x["query_id"]))
            print(
                f"{r['query_id']} spread={float(str(r['spread'])):.1f}x "
                + " ".join(
                    f"{s}={float(str(r[f'{s}_slowdown'])):.2f}"
                    for s in (*STRATEGIES, "duckdb")
                ),
                flush=True,
            )
            _write("imdb_plan_injection.csv", rows)
    summary = summarise(rows)
    _write("imdb_plan_injection_summary.csv", summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
