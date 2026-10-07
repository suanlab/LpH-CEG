"""WCOJ end-to-end experiment: variable ordering quality + execution time.

Compares LP-guided vs standard H-CEG vs baseline orderings on:
1. Residual bound product (ordering quality)
2. Actual WCOJ execution time (GenericJoin simulation)
3. Intermediate result sizes
"""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.utils.query import cycle_query, triangle_query
from src.utils.types import AttributeName, QueryHypergraph, Relation
from src.wcoj.evaluation import (
    lexicographic_ordering,
    max_degree_ordering,
    min_degree_ordering,
)
from src.wcoj.lp_variable_order import lp_variable_ordering
from src.wcoj.variable_order import variable_ordering

if TYPE_CHECKING:
    from pathlib import Path


def generate_power_law_edges(
    n_vertices: int,
    n_edges: int,
    *,
    seed: int,
) -> list[tuple[str, str]]:
    """Generate edges with power-law degree distribution."""
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) for i in range(n_vertices)]
    total = sum(weights)
    probs = [w / total for w in weights]

    def pick() -> str:
        r = rng.random()
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if r <= c:
                return str(i)
        return str(n_vertices - 1)

    tuples: set[tuple[str, str]] = set()
    while len(tuples) < n_edges:
        u, v = pick(), pick()
        if u != v:
            tuples.add((u, v))
    return sorted(tuples)


@dataclass
class WCOJRun:
    """Result of a single GenericJoin execution."""

    ordering: list[str]
    result_count: int
    total_iterations: int
    elapsed_ms: float


def generic_join(
    query: QueryHypergraph,
    relation_data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]],
    ordering: list[AttributeName],
) -> WCOJRun:
    """Simulate GenericJoin with the given variable ordering.

    This is a simplified but faithful implementation of GenericJoin:
    for each prefix length, extend by one variable using intersection
    of candidate values from all relevant relations.
    """
    t0 = time.perf_counter()

    # Build indexes: for each relation, map (attr_name -> index_in_tuple)
    rel_schemas: dict[str, dict[str, int]] = {}
    rel_tuples: dict[str, list[tuple[str, ...]]] = {}
    for rel in query.relations:
        name = rel.name
        attrs, tuples = relation_data[name]
        rel_schemas[name] = {a: i for i, a in enumerate(attrs)}
        rel_tuples[name] = tuples

    # GenericJoin: iteratively extend partial assignments
    partial_results: list[dict[str, str]] = [{}]
    total_iterations = 0

    for var in ordering:
        # Find relations that contain this variable AND at least one bound var
        relevant_rels: list[str] = []
        for rel in query.relations:
            if var in rel.attributes:
                relevant_rels.append(rel.name)

        new_partial: list[dict[str, str]] = []

        for assignment in partial_results:
            # For each relevant relation, find candidate values for var
            candidate_sets: list[set[str]] = []

            for rname in relevant_rels:
                schema = rel_schemas[rname]
                if var not in schema:
                    continue
                var_idx = schema[var]

                # Filter tuples matching current assignment
                candidates: set[str] = set()
                for tup in rel_tuples[rname]:
                    match = True
                    for bound_var, bound_val in assignment.items():
                        if bound_var in schema and tup[schema[bound_var]] != bound_val:
                            match = False
                            break
                    if match:
                        candidates.add(tup[var_idx])
                        total_iterations += 1

                candidate_sets.append(candidates)

            # Intersect all candidate sets
            if candidate_sets:
                intersection = candidate_sets[0]
                for cs in candidate_sets[1:]:
                    intersection = intersection & cs
            else:
                intersection = set()

            # Extend partial results
            for val in intersection:
                extended = dict(assignment)
                extended[var] = val
                new_partial.append(extended)

        partial_results = new_partial

    elapsed = (time.perf_counter() - t0) * 1000

    return WCOJRun(
        ordering=ordering,
        result_count=len(partial_results),
        total_iterations=total_iterations,
        elapsed_ms=elapsed,
    )


def run_experiment(
    label: str,
    query: QueryHypergraph,
    data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]],
    csv_rows: list[dict[str, object]] | None = None,
) -> None:
    """Run all orderings and compare.

    If ``csv_rows`` is provided, one dict per (query, strategy) is appended
    with keys: query, strategy, iterations, time_ms, result_size.
    """
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")

    strategies: dict[str, list[AttributeName]] = {}

    # Standard orderings
    strategies["lexicographic"] = lexicographic_ordering(query)
    strategies["min_degree"] = min_degree_ordering(query)
    strategies["max_degree"] = max_degree_ordering(query)

    # H-CEG guided
    t0 = time.perf_counter()
    strategies["hceg"] = variable_ordering(query, data, max_block_size=1)
    hceg_plan_ms = (time.perf_counter() - t0) * 1000

    # LpH-CEG guided
    t0 = time.perf_counter()
    strategies["lph_ceg"] = lp_variable_ordering(
        query,
        data,
        max_block_size=1,
        p_values=(1.0, 2.0, float("inf")),
    )
    lph_plan_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  Planning time: H-CEG={hceg_plan_ms:.0f}ms, LpH-CEG={lph_plan_ms:.0f}ms")

    # Execute GenericJoin with each ordering
    print(
        f"\n  {'Strategy':<16} {'Ordering':<25} {'|Result|':>10} {'Iterations':>12} {'Time(ms)':>10}"
    )
    print(f"  {'-' * 16} {'-' * 25} {'-' * 10} {'-' * 12} {'-' * 10}")

    results: dict[str, WCOJRun] = {}
    for name, ordering in strategies.items():
        run = generic_join(query, data, ordering)
        results[name] = run
        ord_str = "→".join(ordering)
        if len(ord_str) > 24:
            ord_str = ord_str[:21] + "..."
        print(
            f"  {name:<16} {ord_str:<25} {run.result_count:>10} {run.total_iterations:>12} {run.elapsed_ms:>10.1f}"
        )
        if csv_rows is not None:
            csv_rows.append(
                {
                    "query": label,
                    "strategy": name,
                    "iterations": run.total_iterations,
                    "time_ms": f"{run.elapsed_ms:.4f}",
                    "result_size": run.result_count,
                }
            )

    # Compare LpH-CEG vs H-CEG
    if "hceg" in results and "lph_ceg" in results:
        hceg_r = results["hceg"]
        lph_r = results["lph_ceg"]
        iter_ratio = hceg_r.total_iterations / max(lph_r.total_iterations, 1)
        time_ratio = hceg_r.elapsed_ms / max(lph_r.elapsed_ms, 0.01)
        print(
            f"\n  LpH-CEG vs H-CEG: iterations {iter_ratio:.2f}x, time {time_ratio:.2f}x"
        )
        if results["hceg"].ordering != results["lph_ceg"].ordering:
            print(
                f"  Orderings differ: H-CEG={results['hceg'].ordering}, LpH-CEG={results['lph_ceg'].ordering}"
            )
        else:
            print("  Orderings identical")

    # Find best
    best = min(results, key=lambda n: results[n].total_iterations)
    print(f"  Best strategy: {best}")


def main() -> None:
    print("WCOJ End-to-End Experiment: Variable Ordering → Execution Time")
    print("=" * 65)

    csv_rows: list[dict[str, object]] = []
    seed = 42

    # --- Cycle-4 (small, for detailed analysis) ---
    n_v, n_e = 100, 400
    q = cycle_query(4)
    data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {}
    for i, rel in enumerate(q.relations):
        attrs = tuple(sorted(rel.attributes))
        data[rel.name] = (attrs, generate_power_law_edges(n_v, n_e, seed=seed + i * 7))
    run_experiment(f"Cycle-4 (power-law, {n_v}V, {n_e}E)", q, data, csv_rows)

    # --- Cycle-5 ---
    q5 = cycle_query(5)
    data5: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {}
    for i, rel in enumerate(q5.relations):
        attrs = tuple(sorted(rel.attributes))
        data5[rel.name] = (attrs, generate_power_law_edges(n_v, n_e, seed=seed + i * 7))
    run_experiment(f"Cycle-5 (power-law, {n_v}V, {n_e}E)", q5, data5, csv_rows)

    # --- Cycle-4 (larger) ---
    n_v2, n_e2 = 200, 1000
    data_lg: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {}
    for i, rel in enumerate(q.relations):
        attrs = tuple(sorted(rel.attributes))
        data_lg[rel.name] = (
            attrs,
            generate_power_law_edges(n_v2, n_e2, seed=seed + i * 7),
        )
    run_experiment(f"Cycle-4 (power-law, {n_v2}V, {n_e2}E)", q, data_lg, csv_rows)

    # --- Hetionet Cycle-4 ---
    from pathlib import Path

    hetio = Path(__file__).resolve().parent.parent / "data" / "hetionet"
    if hetio.exists():

        def load_tsv(p: Path, max_r: int = 3000) -> list[tuple[str, str]]:
            edges: list[tuple[str, str]] = []
            with open(p) as f:
                f.readline()
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        edges.append((parts[0], parts[1]))
                        if len(edges) >= max_r:
                            break
            return edges

        ctd = load_tsv(hetio / "ctd_edges.tsv")
        dag = load_tsv(hetio / "dag_edges.tsv")
        cbg = load_tsv(hetio / "cbg_edges.tsv")
        gig = load_tsv(hetio / "gig_edges.tsv")

        q_h4 = QueryHypergraph(
            relations=(
                Relation("CtD", frozenset({"compound", "disease"}), len(ctd)),
                Relation("DaG", frozenset({"disease", "gene1"}), len(dag)),
                Relation("GiG", frozenset({"gene1", "gene2"}), len(gig)),
                Relation("CbG", frozenset({"compound", "gene2"}), len(cbg)),
            ),
            name="hetionet_cycle4",
        )
        data_h4 = {
            "CtD": (("compound", "disease"), ctd),
            "DaG": (("disease", "gene1"), dag),
            "GiG": (("gene1", "gene2"), gig),
            "CbG": (("compound", "gene2"), cbg),
        }
        run_experiment("Hetionet Cycle-4 (heterogeneous)", q_h4, data_h4, csv_rows)

    # --- Triangle (acyclic-like, LP should not help) ---
    q_tri = triangle_query()
    tri_data = {
        "R": (("a", "b"), generate_power_law_edges(n_v, n_e, seed=seed)),
        "S": (("b", "c"), generate_power_law_edges(n_v, n_e, seed=seed + 1)),
        "T": (("c", "a"), generate_power_law_edges(n_v, n_e, seed=seed + 2)),
    }
    run_experiment(
        f"Triangle (power-law, {n_v}V, {n_e}E, control)", q_tri, tri_data, csv_rows
    )

    # --- Write CSV artifact ---
    out_path = (
        Path(__file__).resolve().parent.parent / "results" / "wcoj" / "wcoj_results.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query", "strategy", "iterations", "time_ms", "result_size"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nWCOJ results written to {out_path} ({len(csv_rows)} rows)")

    print("\n" + "=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
