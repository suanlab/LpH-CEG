"""Compare LpH-CEG vs standard H-CEG on synthetic workloads.

Generates power-law triangle/star/path queries and compares:
- CEG (binary, max-degree)
- H-CEG (multi-way, max-degree)
- LpH-CEG (multi-way, LP-optimized Lp-norm)
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING

from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.utils.query import path_query, star_query, triangle_query

if TYPE_CHECKING:
    from src.utils.types import QueryHypergraph


def generate_power_law_relation(
    name: str,
    attrs: tuple[str, ...],
    n_vertices: int,
    n_tuples: int,
    *,
    seed: int,
) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Generate a relation with power-law value distribution."""
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) for i in range(n_vertices)]
    total = sum(weights)
    probs = [w / total for w in weights]

    def pick() -> str:
        r = rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return str(i)
        return str(n_vertices - 1)

    tuples: set[tuple[str, ...]] = set()
    while len(tuples) < n_tuples:
        row = tuple(pick() for _ in attrs)
        if len(set(row)) == len(row):  # no self-loops
            tuples.add(row)

    return attrs, sorted(tuples)


def count_true_result(
    query: QueryHypergraph,
    relation_data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]],
) -> int:
    """Brute-force join to get true cardinality."""
    rels = list(query.relations)
    if not rels:
        return 0

    first = rels[0]
    attrs_0, tuples_0 = relation_data[first.name]
    current = [dict(zip(attrs_0, t, strict=False)) for t in tuples_0]

    for rel in rels[1:]:
        attrs_r, tuples_r = relation_data[rel.name]
        next_tuples = [dict(zip(attrs_r, t, strict=False)) for t in tuples_r]

        shared = set(attrs_0) & set(attrs_r) if current else set()
        # Find shared attributes dynamically
        if current:
            current_attrs = set(current[0].keys())
            shared = current_attrs & set(attrs_r)

        joined: list[dict[str, str]] = []
        # Build index on next_tuples
        idx: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        shared_sorted = sorted(shared)
        for nt in next_tuples:
            key = tuple(nt[a] for a in shared_sorted)
            idx[key].append(nt)

        for ct in current:
            key = tuple(ct[a] for a in shared_sorted)
            for nt in idx.get(key, []):
                merged = {**ct, **nt}
                joined.append(merged)

        current = joined

    return len(current)


def run_comparison(
    query: QueryHypergraph,
    relation_data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]],
    label: str,
) -> None:
    """Run and print comparison for one query."""
    true_count = count_true_result(query, relation_data)

    # CEG (binary, bs=1)
    t0 = time.perf_counter()
    ceg = estimate_pessimistic(query, relation_data, max_block_size=1)
    t_ceg = time.perf_counter() - t0

    # H-CEG (multi-way, bs=2)
    t0 = time.perf_counter()
    hceg = estimate_pessimistic(query, relation_data, max_block_size=2)
    t_hceg = time.perf_counter() - t0

    # LpH-CEG (multi-way + LP, bs=2)
    t0 = time.perf_counter()
    lph = estimate_lph_ceg(
        query,
        relation_data,
        max_block_size=2,
        p_values=(1.0, 2.0, float("inf")),
    )
    t_lph = time.perf_counter() - t0

    def qerror(est: float, true: int) -> float:
        if true == 0:
            return float("inf") if est > 0 else 1.0
        return max(est / true, true / est)

    print(f"\n--- {label} ---")
    print(f"  True cardinality: {true_count}")
    print(
        f"  CEG  (bs=1):  {ceg.estimate:>12.1f}  q-error: {qerror(ceg.estimate, true_count):>8.1f}x  ({t_ceg * 1000:.1f}ms)"
    )
    print(
        f"  H-CEG(bs=2):  {hceg.estimate:>12.1f}  q-error: {qerror(hceg.estimate, true_count):>8.1f}x  ({t_hceg * 1000:.1f}ms)"
    )
    print(
        f"  LpH  (bs=2):  {lph.estimate:>12.1f}  q-error: {qerror(lph.estimate, true_count):>8.1f}x  ({t_lph * 1000:.1f}ms)"
    )

    variant = lph.metadata.get("method_variant", "?")
    ratio = lph.metadata.get("improvement_ratio", 1.0)
    p_used = lph.metadata.get("lp_p_values_used", [])
    print(f"  LP variant: {variant}, improvement: {ratio:.2f}x, p-used: {p_used}")


def main() -> None:
    print("=" * 70)
    print("LpH-CEG vs H-CEG vs CEG Comparison")
    print("=" * 70)

    n_v = 200
    n_e = 800
    seed = 42

    # --- Triangle query ---
    q_tri = triangle_query()
    tri_data = {
        "R": generate_power_law_relation("R", ("a", "b"), n_v, n_e, seed=seed),
        "S": generate_power_law_relation("S", ("b", "c"), n_v, n_e, seed=seed + 1),
        "T": generate_power_law_relation("T", ("c", "a"), n_v, n_e, seed=seed + 2),
    }
    run_comparison(q_tri, tri_data, f"Triangle ({n_v}V, {n_e}E)")

    # --- Path-3 query ---
    q_p3 = path_query(3)
    p3_data = {
        "R1": generate_power_law_relation("R1", ("a1", "a2"), n_v, n_e, seed=seed + 10),
        "R2": generate_power_law_relation("R2", ("a2", "a3"), n_v, n_e, seed=seed + 11),
        "R3": generate_power_law_relation("R3", ("a3", "a4"), n_v, n_e, seed=seed + 12),
    }
    run_comparison(q_p3, p3_data, f"Path-3 ({n_v}V, {n_e}E)")

    # --- Star-3 query --- (uses R(b,a), S1(c1,a), S2(a,c2), S3(c3,a))
    q_s3 = star_query(3)
    s3_data = {
        "R": generate_power_law_relation("R", ("b", "a"), n_v, n_e, seed=seed + 20),
        "S1": generate_power_law_relation("S1", ("c1", "a"), n_v, n_e, seed=seed + 21),
        "S2": generate_power_law_relation("S2", ("a", "c2"), n_v, n_e, seed=seed + 22),
        "S3": generate_power_law_relation("S3", ("c3", "a"), n_v, n_e, seed=seed + 23),
    }
    run_comparison(q_s3, s3_data, f"Star-3 ({n_v}V, {n_e}E)")

    # --- Larger triangle ---
    n_v2, n_e2 = 500, 3000
    tri_data2 = {
        "R": generate_power_law_relation("R", ("a", "b"), n_v2, n_e2, seed=seed + 30),
        "S": generate_power_law_relation("S", ("b", "c"), n_v2, n_e2, seed=seed + 31),
        "T": generate_power_law_relation("T", ("c", "a"), n_v2, n_e2, seed=seed + 32),
    }
    run_comparison(q_tri, tri_data2, f"Triangle ({n_v2}V, {n_e2}E)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
