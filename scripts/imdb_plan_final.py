"""Join ordering on IMDb with the paper's own estimator, repeated, run alone.

``imdb_plan_injection.py`` chose orders with a three-norm flow candidate and
timed every order once, ten queries at a time.  This reruns the comparison
with the estimators the headline numbers use and a repeated timing protocol:

  lpbound_p11  LpBound's bound on single-relation statistics, eleven norms
               (its flow LP, which reaches the entropic optimum at bs=1)
  block_p11    the released guarantee: ``guaranteed_block_bound`` over the
               bs<=2 catalogue, eleven norms (the polymatroid optimum at these
               sizes), statistics from the sparse collector
  oracle       exact prefix counts (a reference; its planning is a join)
  duckdb       DuckDB's own optimizer

Each strategy picks the connected left-deep order minimising C_out, the sum
of its estimates over the order's prefixes.  Planning time (statistics and
solves for every prefix the strategy needs) is recorded per strategy.  The
chosen orders and SAMPLE further sampled orders are timed REPS times each,
one query at a time, every repetition in its own shuffled order; an order
that hits TIMEOUT_S in the first repetition is recorded at the budget and not
repeated.  Slowdowns use the per-order median over repetitions against the
best timed order.  The server is shared: each query records the 1-minute load.

Outputs: results/imdb/imdb_plan_final.csv and _summary.csv.
"""

from __future__ import annotations

import csv
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.e2e_plan_injection import build_sql, connected_orders, load_duckdb
from scripts.evaluate_imdb_suite import build_imdb_query, verify_registration
from scripts.imdb_plan_injection import _sub, run_timed
from scripts.pset_fair import P11
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import (
    block_constraints,
    flow_bound,
    guaranteed_block_bound,
)
from src.estimators.lph_ceg import estimate_lph_ceg
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.statistics.pushdown_stats import compute_sparse_block_extensions

if TYPE_CHECKING:
    import duckdb

    from src.statistics.block_stats import RelationDataMap
    from src.utils.types import QueryHypergraph

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "imdb"
TIMEOUT_S = 60.0
SAMPLE = 16
REPS = 5
DUCKDB_THREADS = 4
SEED = 2027
STRATEGIES = ("lpbound_p11", "block_p11", "oracle")


def _lpbound(sub: QueryHypergraph, data: RelationDataMap) -> float:
    attrs = sorted(sub.all_attributes)
    singles = enumerate_blocks(sub, max_block_size=1)
    st = compute_all_lp_block_extensions(sub, singles, data, p_values=P11)
    cons = block_constraints(sub, data, max_block_size=1, p_values=P11, lp_stats=st)
    return flow_bound(attrs, cons).bound


def _block(sub: QueryHypergraph, data: RelationDataMap) -> float:
    attrs = sorted(sub.all_attributes)
    blocks = enumerate_blocks(sub, max_block_size=2)
    singles = [b for b in blocks if len(b.relations) == 1]
    st = dict(compute_all_lp_block_extensions(sub, singles, data, p_values=P11))
    for b in blocks:
        if len(b.relations) == 2:
            st.update(compute_sparse_block_extensions(sub, b, data, p_values=P11))
    cover = estimate_lph_ceg(
        sub, data, max_block_size=2, p_values=P11, lp_stats=st
    ).estimate
    cons = block_constraints(sub, data, max_block_size=2, p_values=P11, lp_stats=st)
    g = guaranteed_block_bound(attrs, cons, None, math.log2(max(cover, 1e-300)))
    return float(2.0**g.log_bound)


def _oracle(
    sub: QueryHypergraph, names: tuple[str, ...], con: duckdb.DuckDBPyConnection
) -> float:
    con.execute("SET disabled_optimizers=''")
    return float(con.execute(build_sql(sub, names)).fetchone()[0])  # type: ignore[index]


def case(q: dict) -> dict[str, object]:  # type: ignore[type-arg]
    query, data = build_imdb_query(q)
    con = load_duckdb(query, data)
    con.execute(f"SET threads TO {DUCKDB_THREADS}")
    orders = connected_orders(query)
    load_before = os.getloadavg()[0]
    est: dict[str, dict[tuple[str, ...], float]] = {s: {} for s in STRATEGIES}
    plan_s = dict.fromkeys(STRATEGIES, 0.0)

    def estimate(s: str, prefix: tuple[str, ...]) -> float:
        key = tuple(sorted(prefix))
        if key not in est[s]:
            sub = _sub(query, prefix)
            t0 = time.perf_counter()
            if s == "lpbound_p11":
                est[s][key] = _lpbound(sub, data)
            elif s == "block_p11":
                est[s][key] = _block(sub, data)
            else:
                est[s][key] = _oracle(sub, prefix, con)
            plan_s[s] += time.perf_counter() - t0
        return est[s][key]

    def cout(o: tuple[str, ...], s: str) -> float:
        return sum(estimate(s, o[:k]) for k in range(2, len(o) + 1))

    chosen = {s: min(orders, key=lambda o, s=s: cout(o, s)) for s in STRATEGIES}
    rng = random.Random(f"{SEED}:{q['query_id']}")
    sample = rng.sample(orders, min(SAMPLE, len(orders)))
    timed_orders = sorted(dict.fromkeys([*chosen.values(), *sample]))
    runs: dict[object, list[float]] = {o: [] for o in timed_orders}
    runs["duckdb"] = []
    timed_out: set[object] = set()
    counts: set[int] = set()
    for rep in range(REPS):
        todo: list[object] = [o for o in timed_orders if o not in timed_out]
        todo.append("duckdb")
        rng.shuffle(todo)
        for o in todo:
            if o == "duckdb":
                c, ms, to = run_timed(con, build_sql(query, chosen["block_p11"]), False)
            else:
                c, ms, to = run_timed(con, build_sql(query, o), True)  # type: ignore[arg-type]
            runs[o].append(ms)
            if to and rep == 0:
                timed_out.add(o)
            if c is not None and not to:
                counts.add(c)
    con.close()
    assert len(counts) <= 1, f"{q['query_id']}: orders disagree {counts}"
    med = {o: statistics.median(v) for o, v in runs.items()}
    forced = [o for o in timed_orders]
    best = min(med[o] for o in forced)
    row: dict[str, object] = {
        "query_id": q["query_id"],
        "n_orders": len(orders),
        "n_timed": len(forced),
        "reps": REPS,
        "count": next(iter(counts)) if counts else "",
        "best_ms": f"{best:.2f}",
        "n_timeouts": sum(o in timed_out for o in forced),
        "load1": f"{load_before:.1f}",
    }
    for s, o in chosen.items():
        row[f"{s}_slowdown"] = f"{med[o] / best:.4f}"
        row[f"{s}_timeout"] = int(o in timed_out)
        row[f"{s}_plan_s"] = f"{plan_s[s]:.3f}"
        row[f"{s}_exec_ms"] = f"{med[o]:.2f}"
        row[f"{s}_cv"] = (
            f"{statistics.pstdev(runs[o]) / statistics.mean(runs[o]):.3f}"
            if len(runs[o]) > 1
            else ""
        )
    row["duckdb_slowdown"] = f"{med['duckdb'] / best:.4f}"
    row["duckdb_exec_ms"] = f"{med['duckdb']:.2f}"
    row["duckdb_timeout"] = int("duckdb" in timed_out)
    row["block_eq_oracle_order"] = int(chosen["block_p11"] == chosen["oracle"])
    row["block_eq_lpbound_order"] = int(chosen["block_p11"] == chosen["lpbound_p11"])
    return row


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def q95(v: list[float]) -> float:
        v = sorted(v)
        return v[min(len(v) - 1, int(0.95 * len(v)))]

    done = [r for r in rows if int(str(r["n_timeouts"])) == 0]
    out: list[dict[str, object]] = []
    for s in (*STRATEGIES, "duckdb"):
        v = [float(str(r[f"{s}_slowdown"])) for r in done]
        rec: dict[str, object] = {
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
        }
        if s != "duckdb":
            rec["median_plan_s"] = (
                f"{statistics.median(float(str(r[f'{s}_plan_s'])) for r in done):.4g}"
            )
            # planning plus the chosen order's execution, against DuckDB's
            rec["median_total_over_duckdb"] = (
                f"{statistics.median((float(str(r[f'{s}_plan_s'])) * 1000 + float(str(r[f'{s}_exec_ms']))) / float(str(r['duckdb_exec_ms'])) for r in done):.4g}"
            )
            rec["median_cv"] = (
                f"{statistics.median(float(str(r[f'{s}_cv'])) for r in done if str(r[f'{s}_cv'])):.3g}"
            )
        if s == "block_p11":
            rec["n_eq_oracle_order"] = sum(
                int(str(r["block_eq_oracle_order"])) for r in rows
            )
            rec["n_eq_lpbound_order"] = sum(
                int(str(r["block_eq_lpbound_order"])) for r in rows
            )
            # paired: queries where the block order is faster / slower than
            # LpBound's (median over repetitions), ties when equal orders
            rec["n_block_faster_than_lpbound"] = sum(
                float(str(r["block_p11_slowdown"]))
                < float(str(r["lpbound_p11_slowdown"])) * 0.95
                for r in done
            )
            rec["n_block_slower_than_lpbound"] = sum(
                float(str(r["block_p11_slowdown"]))
                > float(str(r["lpbound_p11_slowdown"])) * 1.05
                for r in done
            )
        out.append(rec)
    return out


def _write(name: str, rows: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    if "--summarise" in sys.argv:
        with open(OUT / "imdb_plan_final.csv") as fh:
            _write("imdb_plan_final_summary.csv", summarise(list(csv.DictReader(fh))))
        return
    reg = verify_registration()
    qs = [q for q in reg["queries"] if q["n_vars"] == 6]
    only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
    if only:
        qs = [q for q in qs if q["query_id"] in only]
    rows: list[dict[str, object]] = []
    for q in qs:  # one query at a time: no concurrent timing
        r = case(q)
        rows.append(r)
        print(
            f"{r['query_id']} "
            + " ".join(
                f"{s}={float(str(r[f'{s}_slowdown'])):.2f}"
                for s in (*STRATEGIES, "duckdb")
            ),
            flush=True,
        )
        if not only:
            _write("imdb_plan_final.csv", rows)
    if only:
        return
    summary = summarise(rows)
    _write("imdb_plan_final_summary.csv", summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
