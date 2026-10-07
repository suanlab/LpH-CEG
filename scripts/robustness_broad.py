"""C4: distributional tail-robustness of bound-guided join ordering (>=30 queries).

The robustness result (Table robust) rests on 9 queries. This study widens
it to a distribution: across many cyclic/clique queries it measures each
strategy's slowdown over the best evaluated plan, then reports the median
and the tail (P95). The pre-registered C4 bar (TOP_PLAN): P95 slowdown of
bound-guided << P95 of DuckDB's optimizer, with bound-guided median <= 1.5x.

Reuses the plan-injection harness (DuckDB, join-order optimizer disabled).
Intractable joins are censored via a cheap cover-bound pre-filter.

Output: results/e2e/robustness_broad.csv (+ _summary.csv)
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from scripts.e2e_plan_injection import (
    build_sql,
    choose_order,
    clique_edges,
    connected_orders,
    gen_edges,
    load_duckdb,
    time_execution,
)
from src.estimators.agm_bound import estimate_agm
from src.utils.types import QueryHypergraph, Relation

RelData = dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]


def build_seeded(
    edges: list[tuple[str, str]], nv: int, ne: int, seed: int
) -> tuple[QueryHypergraph, RelData]:
    """Seed-aware graph builder (the shared build_graph fixes its seed)."""
    rels: list[Relation] = []
    data: RelData = {}
    for i, (u, v) in enumerate(edges):
        rows = gen_edges(nv, ne, seed * 1000 + i * 7)
        attrs = tuple(sorted((u, v)))
        swap = attrs != (u, v)
        rels.append(Relation(f"R{i}", frozenset({u, v}), len(rows)))
        data[f"R{i}"] = (attrs, [(y, x) for x, y in rows] if swap else rows)
    return QueryHypergraph(relations=tuple(rels), name="q"), data


P = (1.0, 2.0, float("inf"))
CAP = 300_000  # censor joins too large to enumerate repeatedly


def cyc(k: int) -> list[tuple[str, str]]:
    v = [chr(ord("a") + i) for i in range(k)]
    return [(v[i], v[(i + 1) % k]) for i in range(k)]


SHAPES = {
    "cycle3": cyc(3),
    "cycle4": cyc(4),
    "cycle5": cyc(5),
    "k4": clique_edges("abcd"),
    "diamond": [*cyc(4), ("a", "c")],
    "chordal5": [*cyc(5), ("a", "c")],
}


def pctl(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def main() -> None:
    print("C4: broad tail-robustness of bound-guided ordering")
    print("=" * 70)

    suite = []
    for shape, edges in SHAPES.items():
        for nv, ne in ((80, 300), (110, 480)):
            for seed in (1, 2, 3):
                suite.append((shape, edges, nv, ne, seed))

    rng = random.Random(0)
    rows, censored = [], 0
    for shape, edges, nv, ne, seed in suite:
        q, data = build_seeded(edges, nv, ne, seed)
        # Censor on the AGM bound, which depends only on relation sizes and
        # is therefore independent of BOTH strategies being compared.  Using
        # the proposed method's own bound here would drop exactly the queries
        # where that bound is loose, biasing the comparison in its favour.
        if estimate_agm(q, data) > CAP:
            censored += 1
            continue
        orders = connected_orders(q)
        bound_order = choose_order(q, data, orders, worst=False)
        sample = orders if len(orders) <= 16 else rng.sample(orders, 16)
        eval_orders = list({bound_order, *sample})
        con = load_duckdb(q, data)
        counts = set()
        forced = {}
        for o in eval_orders:
            c, ms = time_execution(con, build_sql(q, o), force_order=True, runs=1)
            counts.add(c)
            forced[o] = ms
        bc, bound_ms = time_execution(
            con, build_sql(q, bound_order), force_order=True, runs=3
        )
        counts.add(bc)
        forced[bound_order] = bound_ms
        dc, duckdb_ms = time_execution(
            con, build_sql(q, bound_order), force_order=False, runs=3
        )
        counts.add(dc)
        con.close()
        if len(counts) != 1:
            continue  # skip inconsistent (shouldn't happen)
        best = min(forced.values())
        rows.append(
            {
                "shape": shape,
                "nv": nv,
                "ne": ne,
                "seed": seed,
                "count": counts.pop(),
                "best_ms": f"{best:.3f}",
                "bound_slowdown": bound_ms / best,
                "duckdb_slowdown": duckdb_ms / best,
            }
        )

    n = len(rows)
    bg = [r["bound_slowdown"] for r in rows]
    dk = [r["duckdb_slowdown"] for r in rows]
    print(f"  queries: {n} ({censored} censored)")
    print(f"  {'strategy':<16} {'median':>8} {'P90':>8} {'P95':>8} {'max':>8}")
    print(f"  {'-' * 50}")
    summ = []
    for name, xs in (("bound-guided", bg), ("DuckDB-opt", dk)):
        med, p90, p95, mx = pctl(xs, 0.5), pctl(xs, 0.9), pctl(xs, 0.95), max(xs)
        print(f"  {name:<16} {med:>7.2f}x {p90:>7.2f}x {p95:>7.2f}x {mx:>7.2f}x")
        summ.append(
            {
                "strategy": name,
                "n": n,
                "median": f"{med:.3f}",
                "p90": f"{p90:.3f}",
                "p95": f"{p95:.3f}",
                "max": f"{mx:.3f}",
            }
        )
    bg_p95, dk_p95 = pctl(bg, 0.95), pctl(dk, 0.95)
    bg_med = pctl(bg, 0.5)
    bar = dk_p95 >= 2 * bg_p95 and bg_med <= 1.5
    print(
        f"\n  C4 bar (DuckDB P95 >= 2x bound-guided P95 & bound median <=1.5x): "
        f"{'MET' if bar else 'NOT MET'}"
    )
    print(
        f"    bound P95={bg_p95:.2f}x, DuckDB P95={dk_p95:.2f}x, "
        f"bound median={bg_med:.2f}x"
    )

    out = Path(__file__).resolve().parent.parent / "results" / "e2e"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "robustness_broad.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "shape",
                "nv",
                "ne",
                "seed",
                "count",
                "best_ms",
                "bound_slowdown",
                "duckdb_slowdown",
            ],
        )
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["bound_slowdown"] = f"{r['bound_slowdown']:.4f}"
            r["duckdb_slowdown"] = f"{r['duckdb_slowdown']:.4f}"
            w.writerow(r)
    with open(out / "robustness_broad_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    print(f"\nWrote {out / 'robustness_broad.csv'} ({n} queries)")


if __name__ == "__main__":
    main()
