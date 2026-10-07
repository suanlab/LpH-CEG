"""X2: Robustness of bound-guided join ordering on a real engine (DuckDB).

A cardinality *upper bound* is not a drop-in cost model for binary-join
ordering: it never underestimates, so ranking left-deep plans by the bound
does not reliably beat a modern sampling-based optimizer, and a *tighter*
bound need not yield a faster plan. What a pessimistic bound *does* give is
**robustness**: by construction it cannot be fooled into a catastrophic plan
on the cyclic, skewed queries where sampling-based estimators
underestimate intermediate sizes and the engine optimizer occasionally
picks a disastrous order.

This experiment quantifies that. For each query we:

1. Enumerate connected left-deep join orders and evaluate a sample of them
   in DuckDB with the join-order optimizer **disabled**
   (``SET disabled_optimizers='join_order'``), timing real execution. This
   estimates the achievable range ``[best, worst]`` of plan runtimes.
2. Force the **bound-guided** order (min-``C_out`` under the LpH-CEG cover
   bound) and time it.
3. Run DuckDB's **own optimizer** (join_order ON).

Per query we report each strategy's *slowdown over the best evaluated plan*.
The headline is the **tail across the suite**: bound-guided ordering stays
within a small factor of optimal on every query, whereas the engine
optimizer---excellent on average---has a catastrophic tail on dense cyclic
queries. The output ``count`` is identical across all orders (correctness
check). Output: ``results/e2e/plan_injection.csv``.
"""

from __future__ import annotations

import csv
import random
import statistics
import time
from itertools import combinations, permutations
from pathlib import Path

import duckdb

from src.estimators.lph_ceg import estimate_lph_ceg
from src.utils.query import cycle_query
from src.utils.types import QueryHypergraph, Relation

P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))
N_EXEC_RUNS: int = 3  # median-of-N wall-clock per evaluated order
SAMPLE_CAP: int = 24  # exhaustive if #orders <= cap, else deterministic sample
RelData = dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]


# ---------------------------------------------------------------------------
# Data generation (same deterministic power-law generator as other scripts).
# ---------------------------------------------------------------------------
def gen_edges(n_v: int, n_e: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) for i in range(n_v)]
    total = sum(weights)
    probs = [w / total for w in weights]

    def pick() -> str:
        r = rng.random()
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if r <= c:
                return str(i)
        return str(n_v - 1)

    tuples: set[tuple[str, str]] = set()
    while len(tuples) < n_e:
        u, v = pick(), pick()
        if u != v:
            tuples.add((u, v))
    return sorted(tuples)


def build_synthetic(q: QueryHypergraph, n_v: int, n_e: int) -> RelData:
    data: RelData = {}
    for i, rel in enumerate(q.relations):
        attrs = tuple(sorted(rel.attributes))
        data[rel.name] = (attrs, gen_edges(n_v, n_e, 42 + i * 7))
    return data


def build_graph(
    name: str, edges: list[tuple[str, str]], n_v: int, n_e: int
) -> tuple[QueryHypergraph, RelData]:
    """Build a query from an explicit edge list (one binary relation per edge)."""
    rels: list[Relation] = []
    data: RelData = {}
    for i, (u, v) in enumerate(edges):
        nm = f"R_{u}{v}"
        attrs = tuple(sorted((u, v)))
        rels.append(Relation(nm, frozenset({u, v}), 0))
        data[nm] = (attrs, gen_edges(n_v, n_e, 42 + i * 7))
    return QueryHypergraph(relations=tuple(rels), name=name), data


def clique_edges(verts: str) -> list[tuple[str, str]]:
    return [(u, v) for u, v in combinations(verts, 2)]


# ---------------------------------------------------------------------------
# Sub-query cardinality via the LpH-CEG cover bound (cached per subset).
# ---------------------------------------------------------------------------
def _sub_query(query: QueryHypergraph, names: tuple[str, ...]) -> QueryHypergraph:
    rels = tuple(r for r in query.relations if r.name in names)
    return QueryHypergraph(relations=rels, name="+".join(names))


def estimate_cover(
    query: QueryHypergraph, data: RelData, names: tuple[str, ...]
) -> float:
    return estimate_lph_ceg(
        _sub_query(query, names), data, max_block_size=1, p_values=P_VALUES
    ).estimate


# ---------------------------------------------------------------------------
# Connected left-deep order selection by C_out (sum of intermediate sizes).
# ---------------------------------------------------------------------------
def _attrs_of(query: QueryHypergraph, name: str) -> frozenset[str]:
    for r in query.relations:
        if r.name == name:
            return r.attributes
    raise KeyError(name)


def connected_orders(query: QueryHypergraph) -> list[tuple[str, ...]]:
    """All left-deep orders where every prefix is connected (no cross join)."""
    names = [r.name for r in query.relations]
    out: list[tuple[str, ...]] = []
    for perm in permutations(names):
        covered = set(_attrs_of(query, perm[0]))
        ok = True
        for nm in perm[1:]:
            a = _attrs_of(query, nm)
            if covered.isdisjoint(a):
                ok = False
                break
            covered |= a
        if ok:
            out.append(perm)
    return out


def cout_cost(
    query: QueryHypergraph,
    data: RelData,
    order: tuple[str, ...],
    cache: dict[tuple[str, ...], float],
) -> float:
    """Sum of cover-bound cardinalities of the size-2..k connected prefixes."""
    total = 0.0
    for plen in range(2, len(order) + 1):
        prefix = tuple(sorted(order[:plen]))
        if prefix not in cache:
            cache[prefix] = estimate_cover(query, data, prefix)
        total += cache[prefix]
    return total


def choose_order(
    query: QueryHypergraph, data: RelData, orders: list[tuple[str, ...]], *, worst: bool
) -> tuple[str, ...]:
    cache: dict[tuple[str, ...], float] = {}
    scored = [(cout_cost(query, data, o, cache), o) for o in orders]
    return (max(scored) if worst else min(scored))[1]


# ---------------------------------------------------------------------------
# DuckDB: load data and execute a forced join order.
# ---------------------------------------------------------------------------
def load_duckdb(query: QueryHypergraph, data: RelData) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for rel in query.relations:
        attrs, rows = data[rel.name]
        cols = ", ".join(f"{a} VARCHAR" for a in attrs)
        con.execute(f"CREATE TABLE {rel.name} ({cols})")
        con.executemany(
            f"INSERT INTO {rel.name} VALUES ({', '.join('?' for _ in attrs)})",
            rows,
        )
    return con


def build_sql(query: QueryHypergraph, order: tuple[str, ...]) -> str:
    """Left-deep COUNT(*) SQL whose syntactic order follows ``order``."""
    first = order[0]
    attr_owner: dict[str, str] = {a: first for a in _attrs_of(query, first)}
    clauses = [f"FROM {first}"]
    for nm in order[1:]:
        a = _attrs_of(query, nm)
        conds = [
            f"{nm}.{attr} = {attr_owner[attr]}.{attr}"
            for attr in sorted(a)
            if attr in attr_owner
        ]
        clauses.append(f"JOIN {nm} ON {' AND '.join(conds)}")
        for attr in a:
            attr_owner.setdefault(attr, nm)
    return "SELECT count(*) " + " ".join(clauses)


def time_execution(
    con: duckdb.DuckDBPyConnection, sql: str, *, force_order: bool, runs: int
) -> tuple[int, float]:
    """Return (count, median_ms) executing sql; optionally forcing the order."""
    con.execute(
        "SET disabled_optimizers='join_order'"
        if force_order
        else "SET disabled_optimizers=''"
    )
    count = con.execute(sql).fetchone()[0]
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        con.execute(sql).fetchone()
        samples.append((time.perf_counter() - t0) * 1000)
    return int(count), statistics.median(samples)


# ---------------------------------------------------------------------------
# Robustness driver
# ---------------------------------------------------------------------------
def robustness_case(
    label: str,
    query: QueryHypergraph,
    data: RelData,
    csv_rows: list[dict[str, object]],
) -> dict[str, float]:
    print(f"\n{'=' * 74}\n  {label}\n{'=' * 74}")

    orders = connected_orders(query)
    bound_order = choose_order(query, data, orders, worst=False)

    # Estimate the achievable [best, worst] runtime range by evaluating a
    # bounded, deterministic sample of orders (exhaustive when small).
    rng = random.Random(0)
    sample = orders if len(orders) <= SAMPLE_CAP else rng.sample(orders, SAMPLE_CAP)
    eval_orders = list({bound_order, *sample})

    con = load_duckdb(query, data)
    counts: set[int] = set()
    forced_ms: dict[tuple[str, ...], float] = {}
    # Range estimation only needs a single timed run per sampled order; the
    # two *reported* strategies are re-timed with median-of-N for stability.
    for o in eval_orders:
        c, ms = time_execution(con, build_sql(query, o), force_order=True, runs=1)
        counts.add(c)
        forced_ms[o] = ms
    bc, bound_ms = time_execution(
        con, build_sql(query, bound_order), force_order=True, runs=N_EXEC_RUNS
    )
    counts.add(bc)
    forced_ms[bound_order] = bound_ms
    dc, duckdb_ms = time_execution(
        con, build_sql(query, bound_order), force_order=False, runs=N_EXEC_RUNS
    )
    counts.add(dc)
    con.close()
    assert len(counts) == 1, f"orders disagree on count: {counts}"
    count = counts.pop()

    best = min(forced_ms.values())
    worst = max(forced_ms.values())
    bound_ms = forced_ms[bound_order]
    bound_slow = bound_ms / best
    duckdb_slow = duckdb_ms / best
    worst_slow = worst / best

    print(
        f"  orders={len(orders)} (evaluated {len(eval_orders)})  count={count}\n"
        f"  best={best:.1f}ms  worst={worst:.1f}ms  spread={worst_slow:.2f}x"
    )
    print(
        f"  bound-guided: {bound_ms:7.1f}ms  ({bound_slow:.2f}x of best)\n"
        f"  DuckDB-opt  : {duckdb_ms:7.1f}ms  ({duckdb_slow:.2f}x of best)"
    )
    verdict = (
        "engine catastrophic"
        if duckdb_slow > 1.5 * bound_slow and duckdb_slow > 2
        else "comparable"
    )
    print(f"  -> {verdict}")

    csv_rows.append(
        {
            "query": label,
            "n_orders": len(orders),
            "n_evaluated": len(eval_orders),
            "count": count,
            "best_ms": f"{best:.4f}",
            "worst_ms": f"{worst:.4f}",
            "spread": f"{worst_slow:.4f}",
            "bound_guided_ms": f"{bound_ms:.4f}",
            "bound_guided_slowdown": f"{bound_slow:.4f}",
            "duckdb_ms": f"{duckdb_ms:.4f}",
            "duckdb_slowdown": f"{duckdb_slow:.4f}",
        }
    )
    return {"bound_slow": bound_slow, "duckdb_slow": duckdb_slow}


# ---------------------------------------------------------------------------
# Real-data heterogeneous triangle (large GiG relation -> wide plan spread).
# ---------------------------------------------------------------------------
def load_hetionet_triangle() -> tuple[QueryHypergraph, RelData] | None:
    hetio = Path(__file__).resolve().parent.parent / "data" / "hetionet"
    if not hetio.exists():
        return None

    def load_tsv(p: Path) -> list[tuple[str, str]]:
        edges: list[tuple[str, str]] = []
        with open(p) as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    edges.append((parts[0], parts[1]))
        return edges

    try:
        cbg = load_tsv(hetio / "cbg_edges.tsv")  # compound, gene_a
        gig = load_tsv(hetio / "gig_edges.tsv")  # gene_a, gene_b  (large)
        cdg = load_tsv(hetio / "CdG_edges.tsv")  # compound, gene_b
    except OSError:
        return None

    q = QueryHypergraph(
        relations=(
            Relation("CbG", frozenset({"compound", "gene_a"}), len(cbg)),
            Relation("GiG", frozenset({"gene_a", "gene_b"}), len(gig)),
            Relation("CdG", frozenset({"compound", "gene_b"}), len(cdg)),
        ),
        name="hetionet_triangle",
    )
    data: RelData = {
        "CbG": (("compound", "gene_a"), cbg),
        "CdG": (("compound", "gene_b"), cdg),
        "GiG": (("gene_a", "gene_b"), gig),
    }
    return q, data


def main() -> None:
    print("X2: Robustness of bound-guided join ordering (DuckDB, real execution)")
    print("=" * 74)

    csv_rows: list[dict[str, object]] = []
    summary: list[tuple[str, dict[str, float]]] = []

    def run(label: str, q: QueryHypergraph, d: RelData) -> None:
        summary.append((label, robustness_case(label, q, d, csv_rows)))

    # Pure cycles (proper prefixes acyclic -> bound-guided == max-degree order).
    run(
        "Cycle-4 (100V/400E)", cycle_query(4), build_synthetic(cycle_query(4), 100, 400)
    )
    run(
        "Cycle-5 (100V/400E)", cycle_query(5), build_synthetic(cycle_query(5), 100, 400)
    )

    # Dense cyclic queries with cyclic intermediates -- the robustness regime.
    # Several K4 densities/sizes to check the engine's catastrophic tail is
    # systematic, not a single cherry-picked configuration.
    for nv, ne in [(120, 800), (120, 1500), (120, 2500), (200, 2000)]:
        q, d = build_graph("K4", clique_edges("abcd"), nv, ne)
        run(f"K4 4-clique ({nv}V/{ne}E)", q, d)

    diamond = [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"), ("a", "c")]
    q_dia, d_dia = build_graph("diamond", diamond, 120, 1500)
    run("Diamond / chordal-4 (120V/1500E)", q_dia, d_dia)

    chordal = [
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
        ("d", "e"),
        ("a", "e"),
        ("a", "c"),
    ]
    q_ch, d_ch = build_graph("chordal5", chordal, 120, 1200)
    run("Chordal 5-cycle (120V/1200E)", q_ch, d_ch)

    hetio = load_hetionet_triangle()
    if hetio is not None:
        run("Hetionet hetero-triangle (real)", hetio[0], hetio[1])
    else:
        print("\n  [Hetionet data not found; skipping real-data case]")

    # --- Robustness summary: the tail matters, not the average. ---
    print(
        f"\n{'=' * 74}\n  ROBUSTNESS SUMMARY (slowdown over best evaluated plan)\n{'=' * 74}"
    )
    print(f"  {'query':<34} {'bound-guided':>13} {'DuckDB-opt':>12}")
    print(f"  {'-' * 34} {'-' * 13} {'-' * 12}")
    for label, m in summary:
        print(f"  {label:<34} {m['bound_slow']:>12.2f}x {m['duckdb_slow']:>11.2f}x")
    bound_tail = max(m["bound_slow"] for _, m in summary)
    duckdb_tail = max(m["duckdb_slow"] for _, m in summary)
    print(f"  {'-' * 34} {'-' * 13} {'-' * 12}")
    print(f"  {'WORST-CASE (tail)':<34} {bound_tail:>12.2f}x {duckdb_tail:>11.2f}x")
    print(
        f"\n  Bound-guided ordering stays within {bound_tail:.1f}x of optimal on every "
        f"query;\n  the engine optimizer's tail is {duckdb_tail:.1f}x "
        f"({duckdb_tail / bound_tail:.1f}x worse tail)."
    )

    out_path = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "e2e"
        / "plan_injection.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query",
        "n_orders",
        "n_evaluated",
        "count",
        "best_ms",
        "worst_ms",
        "spread",
        "bound_guided_ms",
        "bound_guided_slowdown",
        "duckdb_ms",
        "duckdb_slowdown",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nWrote {out_path} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
